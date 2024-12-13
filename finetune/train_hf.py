import os

#  launch script with python for single gpu. (python pair_finetuning.py)
#  for multi-gpu do : accelerate launch --config_file sft_cfg.yaml pair_finetuning.py 


from datetime import datetime
import sys
import argparse
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from accelerate import Accelerator
import json
from datasets import Dataset

from train_utils import get_messages

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
parser.add_argument('--train_data_path', default='data/finetune/train.jsonl')
parser.add_argument('--val_data_path', default='data/finetune/val.jsonl')
parser.add_argument('--lora', default=True)
parser.add_argument('--device_map', default=True, action='store_false')
parser.add_argument('--hub_model_name', default=None)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)  # They finetune on 4 gpus with per device batch size of 2
parser.add_argument('--lora_rank', type=int, default=8) 
parser.add_argument('--max_seq_len', type=int, default=16000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--output_dir', type=str, default='/data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/finetuning_outputs/hf/')
args = parser.parse_args()


if torch.cuda.device_count() > 1:
    multi_gpu = True
    accelerator = Accelerator() 
    device_map = None 
else:
    multi_gpu = False
    device_map = 'auto'

output_dir = args.output_dir + f"{args.model.split('/')[-1]}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"


with open(args.train_data_path) as f:
    train_dataset = [json.loads(line) for line in f]
    train_dataset = [{'previous_content':D['file_previous_content'],'current_content':D['file_current_content'],'draft':D['file_draft_from_diff']} for D in train_dataset]
    train_dataset = Dataset.from_list(train_dataset)
    #train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

with open(args.val_data_path) as f:
    eval_dataset = [json.loads(line) for line in f]
    eval_dataset = [{'previous_content':D['file_previous_content'],'current_content':D['file_current_content'],'draft':D['file_draft_from_diff']} for D in eval_dataset]
    eval_dataset = Dataset.from_list(eval_dataset)
    #val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

base_model = args.model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map= device_map, #None, #"auto" if args.device_map else None  to run with acceleatr
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    # Tokenize the prompt
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
        return_tensors=None,
    )
    
    # Initialize labels with -100
    labels = [-100] * len(result["input_ids"])
    
    # Find the indices of <updated_code> and </updated_code>
    start_tag = "<updated_code>"
    end_tag = "</updated_code>"
    
    # Convert the prompt to a string and find the positions of the tags
    prompt_str = tokenizer.decode(result["input_ids"])
    start_idx = prompt_str.find(start_tag)
    end_idx = prompt_str.find(end_tag, start_idx)
    
    if start_idx != -1 and end_idx != -1:
        # Convert character indices to token indices
        start_token_idx = len(tokenizer.encode(prompt_str[:start_idx], add_special_tokens=False))
        end_token_idx = len(tokenizer.encode(prompt_str[:end_idx], add_special_tokens=False))
        
        # Set the labels for the updated code section
        labels[start_token_idx:end_token_idx] = result["input_ids"][start_token_idx:end_token_idx]
    
    result["labels"] = labels
    return result

def generate_and_tokenize_prompt(data_point):

    messages = get_messages(data_point['previous_content'], data_point['draft'], data_point['current_content'])
    
    # if 'codellama' in base_model:
    #     full_seq = '[INST] ' + full_prompt[:-2] + '[/INST]\n' + response
    # else: # code llama does not have tokenizer.apply_chat_template implemented in huggingface 
    full_seq = tokenizer.apply_chat_template(messages, tokenize=False)

    # print(full_seq)
    return tokenize(full_seq)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train()  # Set to training mode
for name, param in model.named_parameters():
    if param.dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128, torch.bfloat16]:
        param.requires_grad = True

if args.lora:
    print('LORA HAPPENING')
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

wandb_project = '' # "Optim-finetune" let it got to the default HF dir
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# if torch.cuda.device_count() > 1:
#     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
#     model.is_parallelizable = True
#     model.model_parallel = True

batch_size = args.batch_size
per_device_train_batch_size = args.batch_size
# gradient_accumulation_steps = batch_size // per_device_train_batch_size

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        # gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        # max_steps=400,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        optim="adamw_torch_fused",
        # SRIJITH eval_strategy for ecco env and evaluation strategy for spin env
        eval_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        logging_steps=args.log_interval,
        eval_steps= 500, #args.log_interval,
        save_steps= 500, #args.log_interval,
        output_dir=output_dir,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,

        # save_total_limit=3,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"{args.model}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)

# if torch.__version__ >= "2" and sys.platform != "win32":
#     print("compiling the model")
#     model = torch.compile(model)

if trainer.accelerator.is_main_process:
    print('Hi from main process!!!')

print('Starting to train')

trainer.train()

try:
    merged_model_temp = model.merge_and_unload()
    merged_model_temp.save_pretrained(f"{output_dir}/checkpoint-final/HF_checkpoint/")
except:
    pass


if trainer.accelerator.is_main_process: # Running on main process only 
    if args.lora:
        merged_model = model.merge_and_unload()
    else:
        merged_model = model 

    #merged_model.push_to_hub(args.hub_model_name)

    if not os.path.exists(f"{output_dir}/checkpoint-final/"):
        os.makedirs(f"{output_dir}/checkpoint-final/")

    print('Saving final model to', f"{output_dir}/checkpoint-final/merged_model.bin")
    torch.save(merged_model.state_dict(), f"{output_dir}/checkpoint-final/merged_model.bin")

    #trainer.tokenizer.push_to_hub(args.hub_model_name)