import torch
from datasets import load_dataset, Dataset

def get_tokens_from_iterator(stream_iterator, tokenizer, target_tokens, dataset_name):
    all_ids = []
    current_tokens = 0
    
    print(f"Collecting ~{target_tokens/1e6:.1f}M tokens...")
    for item in stream_iterator:
        text_content = None
        if dataset_name == "deepmind/code_contests":
            solutions = item.get('solutions', {}).get('solution', [])
            if solutions:
                text_content = "\n".join(solutions)
        elif dataset_name == "meta-math/MetaMathQA":
            query = item.get('query')
            response = item.get('response')
            if query and response:
                text_content = f"Question: {query}\nAnswer: {response}"
        else:
            text_content = item.get('content')

        if text_content:
            ids = tokenizer.encode(text_content)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)
            current_tokens += len(ids) + 1
            if current_tokens >= target_tokens:
                break
                
    print(f"Collected {current_tokens} tokens.")
    return torch.tensor(all_ids, dtype=torch.long)


def create_blocks(data_tensor, block_size):
    total_length = len(data_tensor)
    total_length = (total_length // block_size) * block_size
    input_ids, labels = [], []
    for i in range(0, total_length, block_size):
        chunk = data_tensor[i : i + block_size]
        input_ids.append(chunk)
        labels.append(chunk.clone())
    return {"input_ids": input_ids, "labels": labels}


def prepare_dataset(config, tokenizer):
    dataset_name = config['dataset_name']
    
    if dataset_name == "deepmind/code_contests":
        train_stream = load_dataset(dataset_name, streaming=True, split='train')
        eval_stream = load_dataset(dataset_name, streaming=True, split='valid')
        
        train_data = get_tokens_from_iterator(iter(train_stream), tokenizer, config['train_tokens'], dataset_name)
        eval_data = get_tokens_from_iterator(iter(eval_stream), tokenizer, config['eval_tokens'], dataset_name)

    else:
        stream_dataset = load_dataset(dataset_name, streaming=True, split='train')
        stream_iterator = iter(stream_dataset)
        
        eval_data = get_tokens_from_iterator(stream_iterator, tokenizer, config['eval_tokens'], dataset_name)
        
        train_data = get_tokens_from_iterator(stream_iterator, tokenizer, config['train_tokens'], dataset_name)

    train_blocks = create_blocks(train_data, config['block_size'])
    eval_blocks = create_blocks(eval_data, config['block_size'])

    train_dataset = Dataset.from_dict(train_blocks)
    eval_dataset = Dataset.from_dict(eval_blocks)
    
    return train_dataset, eval_dataset
