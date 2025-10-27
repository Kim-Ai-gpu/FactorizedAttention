import torch
import yaml
import math
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from transformers.trainer_utils import set_seed
from src.data import prepare_dataset
from src.model import monkey_patch_model, create_lora_model

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    all_baseline_ppls = []
    all_patched_ppls = []
    all_improvements = []

    for seed in config['seeds']:
        print(f"\n--- Running Experiment with Seed: {seed} ---")
        set_seed(seed)
        
        run_name = f"{config['wandb_config']['run_name']}_{config['dataset_name'].split('/')[-1]}_seed{seed}"
        wandb.init(
            project=config['wandb_config']['project_name'], 
            name=run_name,
            config=config,
            reinit=True
        )

        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset, eval_dataset = prepare_dataset(config, tokenizer)

        print("--- Preparing Baseline Model with LoRA ---")
        baseline_model = AutoModelForCausalLM.from_pretrained(config['model_name'])
        baseline_model.config.use_cache = False
        lora_baseline_model = create_lora_model(baseline_model, config['lora_config'], is_patched=False)

        print("\n--- Preparing Patched Model with LoRA ---")
        base_model_for_patching = AutoModelForCausalLM.from_pretrained(config['model_name'])
        base_model_for_patching.config.use_cache = False
        patched_model = monkey_patch_model(base_model_for_patching)
        lora_patched_model = create_lora_model(patched_model, config['lora_config_patched'], is_patched=True)

        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f"Found {n_gpu} GPUs.")
        else:
            print("No GPU found, using CPU.")

        dataset_slug = config['dataset_name'].split('/')[-1]
        training_args_config = config['training_args'].copy()
        base_output_dir = training_args_config.pop('output_dir')
        output_dir = f"{base_output_dir}/{dataset_slug}/seed{seed}"

        training_args = TrainingArguments(
            **training_args_config,
            output_dir=output_dir,
            seed=seed
        )

        print("\n--- Training Baseline LoRA Model ---")
        baseline_trainer = Trainer(
            model=lora_baseline_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )
        baseline_trainer.train()
        baseline_eval = baseline_trainer.evaluate()
        baseline_ppl = math.exp(baseline_eval['eval_loss'])

        print("\n--- Training Patched LoRA Model ---")
        patched_trainer = Trainer(
            model=lora_patched_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )
        patched_trainer.train()
        patched_eval = patched_trainer.evaluate()
        patched_ppl = math.exp(patched_eval['eval_loss'])

        print("\n--- Final Results --- ")
        print(f"Seed: {seed}")
        print(f"Baseline PPL: {baseline_ppl:.4f}")
        print(f"Patched PPL: {patched_ppl:.4f}")
        improvement = (baseline_ppl - patched_ppl) / baseline_ppl * 100
        print(f"Improvement: {improvement:.2f}%")

        all_baseline_ppls.append(baseline_ppl)
        all_patched_ppls.append(patched_ppl)
        all_improvements.append(improvement)

        wandb.finish()

    if len(config['seeds']) > 1:
        baseline_ppl_tensor = torch.tensor(all_baseline_ppls)
        patched_ppl_tensor = torch.tensor(all_patched_ppls)
        improvement_tensor = torch.tensor(all_improvements)

        mean_baseline_ppl = torch.mean(baseline_ppl_tensor)
        std_baseline_ppl = torch.std(baseline_ppl_tensor)

        mean_patched_ppl = torch.mean(patched_ppl_tensor)
        std_patched_ppl = torch.std(patched_ppl_tensor)

        mean_improvement = torch.mean(improvement_tensor)
        std_improvement = torch.std(improvement_tensor)

        print("\n--- Overall Results ---")
        print(f"Baseline PPL: {mean_baseline_ppl:.4f} +/- {std_baseline_ppl:.4f}")
        print(f"Patched PPL: {mean_patched_ppl:.4f} +/- {std_patched_ppl:.4f}")
        print(f"Improvement: {mean_improvement:.2f}% +/- {std_improvement:.2f}%")

        summary_run_name = f"Overall_Summary_{config['dataset_name'].split('/')[-1]}"
        wandb.init(
            project=config['wandb_config']['project_name'],
            name=summary_run_name,
            config=config,
            reinit=True
        )
        wandb.log({
            "mean_baseline_ppl": mean_baseline_ppl.item(),
            "std_baseline_ppl": std_baseline_ppl.item(),
            "mean_patched_ppl": mean_patched_ppl.item(),
            "std_patched_ppl": std_patched_ppl.item(),
            "mean_improvement": mean_improvement.item(),
            "std_improvement": std_improvement.item(),
        })
        wandb.finish()

if __name__ == "__main__":
    main()
