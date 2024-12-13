import argparse
from datetime import datetime
import os
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import json
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from train_utils import clean_code, get_messages
from Levenshtein import distance
from transformers import TrainerCallback

def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model using Unsloth')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen2.5-Coder-7B-Instruct",
                      help='Name or path of the pretrained model')
    parser.add_argument('--max_seq_length', type=int, default=131072,
                      help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                      help='Use 4-bit quantization')
    
    # Training configuration
    parser.add_argument('--train_data-path', type=str, default='data/finetune/train.jsonl',
                      help='Path to training data JSONL file')
    parser.add_argument('--val_data-path', type=str, default='data/finetune/val.jsonl',
                      help='Path to validation data JSONL file')
    parser.add_argument('--output-dir', type=str, default="outputs",
                      help='Path to output directory')
    parser.add_argument('--exp-name', type=str, required=True,
                      help='Name for the output model')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Per device training batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                      help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16,
                      help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=16,
                      help='LoRA alpha parameter')
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.exp_name = args.exp_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    # save args to json
    with open(os.path.join(args.output_dir, args.exp_name, "args.json"), "w") as f:
        json.dump(args.__dict__, f)
    
    # Model initialization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    class MetricsCallback(TrainerCallback):
        def compute_metrics(self, eval_preds):
            predictions, labels = eval_preds
            # Decode tokens to text
            pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Exact match
            exact_matches = sum(p == l for p, l in zip(pred_texts, label_texts))
            exact_match_score = exact_matches / len(pred_texts)
            
            # Edit distance
            edit_distances = [distance(p, l) for p, l in zip(pred_texts, label_texts)]
            avg_edit_distance = sum(edit_distances) / len(edit_distances)
            
            return {
                "exact_match": exact_match_score,
                "edit_distance": avg_edit_distance
            }

    def formatting_prompts_func(examples):
        files = examples["files"]
        convos = [
            get_messages(file['previous_content'], file['draft'], file['current_content'])
            for file in files
        ]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    # Data loading
    with open(args.train_data_path) as f:
        train_dataset = [json.loads(line) for line in f]
    train_dataset = [{'previous_content':D['file_previous_content'],'current_content':D['file_current_content'],'draft':D['file_draft_from_diff']} for D in train_dataset]
    train_dataset = Dataset.from_dict({"files": train_dataset})
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    with open(args.val_data_path) as f:
        val_dataset = [json.loads(line) for line in f]
    val_dataset = [{'previous_content':D['file_previous_content'],'current_content':D['file_current_content'],'draft':D['file_draft_from_diff']} for D in val_dataset]
    val_dataset = Dataset.from_dict({"files": val_dataset})
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # Model configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(range(300)),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=4,
        packing=False,
        # compute_metrics=MetricsCallback().compute_metrics,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            per_device_eval_batch_size=3, 
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=os.path.join(args.output_dir, args.exp_name),
            report_to="wandb",
            run_name=args.exp_name,
            save_strategy="steps",
            save_steps=300,
            eval_strategy="steps",
            eval_steps=300,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    class DebugCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            sample_batch = next(iter(trainer.get_train_dataloader()))
        
            # Decode the input IDs 
            input_text = tokenizer.batch_decode(sample_batch['input_ids'], skip_special_tokens=False)
            
            # Handle labels properly - replace -100 with pad token id
            labels = sample_batch['labels'].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            labels_text = tokenizer.batch_decode(labels, skip_special_tokens=False)
            
            print("\n=== TRAINING DATA INSPECTION ===")
            print("\nFull input including special tokens:")
            print(input_text[0][:500])
            
            print("\nLabels (what model is actually training on):")
            print(labels_text[0][:500])
            
            # Find positions where labels aren't -100 to see what's actually being trained on
            training_mask = sample_batch['labels'][0] != -100
            training_ids = sample_batch['input_ids'][0][training_mask]
            training_text = tokenizer.decode(training_ids)
            
            print("\nActual text being trained on (where labels != -100):")
            print(training_text)
            
            # Print the special token IDs for verification
            start_user = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
            end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            start_assistant = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            
            print("\nSpecial token IDs:")
            print(f"<|im_start|>user: {start_user}")
            print(f"<|im_end|>: {end_token}")
            print(f"<|im_start|>assistant: {start_assistant}")
            
            # Print statistics about what's being trained on
            total_tokens = len(training_mask)
            trained_tokens = training_mask.sum().item()
            print(f"\nTraining on {trained_tokens}/{total_tokens} tokens ({trained_tokens/total_tokens*100:.2f}%)")

    # Add the callback to your trainer
    trainer.add_callback(DebugCallback())

    #print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))

    # # Training stats and execution
    # gpu_stats = torch.cuda.get_device_properties(0)
    # start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    # print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # # Final stats
    # used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    # used_percentage = round(used_memory / max_memory * 100, 3)
    # lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    # print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    # print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    # print(f"Peak reserved memory = {used_memory} GB.")
    # print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    # print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    # print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # # Save models
    # model.save_pretrained(os.path.join(args.output_dir, args.exp_name, "adapter"))
    # tokenizer.save_pretrained(
    #     os.path.join(args.output_dir, args.exp_name, "adapter")
    # )
    # model.save_pretrained_merged(
    #     os.path.join(args.output_dir, args.exp_name, f"merged"),
    #     tokenizer,
    #     save_method="merged_16bit",
    # )

if __name__ == "__main__":
    main()
