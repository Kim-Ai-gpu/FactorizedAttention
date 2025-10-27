import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
from peft import get_peft_model, LoraConfig, TaskType

class FactorizedAttention(nn.Module):
    def __init__(self, config, base_attention):
        super().__init__()
        self.config = config
        self.n_head = base_attention.num_heads
        self.head_dim = config.hidden_size // self.n_head
        self.embed_dim = config.hidden_size
        self.rank = 128

        self.c_attn = base_attention.c_attn
        self.c_proj = base_attention.c_proj
        
        # multiplicative factorization
        self.q_w1 = nn.Linear(self.head_dim, self.rank, bias=False)
        self.q_w2 = nn.Linear(self.head_dim, self.rank, bias=False)
        self.k_w3 = nn.Linear(self.head_dim, self.rank, bias=False)
        self.k_w4 = nn.Linear(self.head_dim, self.rank, bias=False)
        
        # projection back to original dimension
        self.q_proj = nn.Linear(self.rank, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.rank, self.head_dim, bias=False)
        
        self.activation = nn.GELU()
        
        # learnable alpha
        self.alpha_q = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        self.alpha_k = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        
        # normalization
        self.pre_norm_q = nn.LayerNorm(self.head_dim)
        self.pre_norm_k = nn.LayerNorm(self.head_dim)
        
        # learnable temperature per head
        self.temperature = nn.Parameter(torch.ones(1, self.n_head, 1, 1))
        
        self.dropout = nn.Dropout(0.05)
        
        self.register_buffer("bias", base_attention.bias)
        self._init_weights()
    
    def _init_weights(self):
        # random initialization
        for m in [self.q_w1, self.q_w2, self.k_w3, self.k_w4]:
            nn.init.orthogonal_(m.weight, gain=0.7)
        
        # projection layers initialization
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        
        nn.init.constant_(self.alpha_q, 0.0)
        nn.init.constant_(self.alpha_k, 0.0)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, 
                head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)
        
        # normalization for stability
        query_norm = self.pre_norm_q(query)
        key_norm = self.pre_norm_k(key)
        
        # multiplicative factorization
        q_f1 = self.q_w1(query_norm)
        q_f2 = self.q_w2(query_norm)
        q_interaction = q_f1 * q_f2  # element wise multiplication
        
        k_f1 = self.k_w3(key_norm)
        k_f2 = self.k_w4(key_norm)
        k_interaction = k_f1 * k_f2
        
        q_struct_raw = self.activation(q_interaction)
        q_struct_raw = self.q_proj(q_struct_raw)
        q_struct_raw = self.dropout(q_struct_raw)
        
        k_struct_raw = self.activation(k_interaction)
        k_struct_raw = self.k_proj(k_struct_raw)
        k_struct_raw = self.dropout(k_struct_raw)
        
        # sigmoid gating
        alpha_q = torch.sigmoid(self.alpha_q)
        alpha_k = torch.sigmoid(self.alpha_k)
        
        # residual connection with learnable mixing
        q_struct = query + alpha_q * q_struct_raw
        k_struct = key + alpha_k * k_struct_raw
        
        # attention matrix computation
        attn_scores = torch.matmul(q_struct, k_struct.transpose(-1, -2))
        attn_scores = attn_scores / (math.sqrt(self.head_dim) * self.temperature)

        # causal masking
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length].bool()
        attn_scores = attn_scores.masked_fill(~causal_mask, torch.finfo(attn_scores.dtype).min)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        attn_output = self.c_proj(attn_output)

        return (attn_output, attn_weights) if output_attentions else (attn_output, None)

def monkey_patch_model(model):
    patched_model = copy.deepcopy(model)
    for layer in patched_model.transformer.h:
        original_attention = layer.attn
        layer.attn = FactorizedAttention(patched_model.config, original_attention)
    return patched_model

def create_lora_model(model, lora_config, is_patched=False):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **lora_config
    )
    lora_model = get_peft_model(model, peft_config)
    
    if is_patched:
        trainable_params = [
            'q_w1', 'q_w2',          
            'k_w3', 'k_w4',          
            'q_proj', 'k_proj',      
            'alpha_q', 'alpha_k',    
            'temperature',           
            'pre_norm_q', 'pre_norm_k',  
            'dropout',                
        ]
        
        factorized_param_count = 0
        for n, p in lora_model.named_parameters():
            if any(param_name in n for param_name in trainable_params):
                p.requires_grad = True
                factorized_param_count += p.numel()
                
        for n, p in lora_model.named_parameters():
            if p.requires_grad and any(param_name in n for param_name in trainable_params):
                print(f"  ✓ {n:50s} | Shape: {str(tuple(p.shape)):20s} | Params: {p.numel():,}")
        print(f"{'='*60}")
        print(f"Total Factorized Attention params: {factorized_param_count:,}")
        print(f"{'='*60}\n")
    
    lora_model.print_trainable_parameters()
    return lora_model