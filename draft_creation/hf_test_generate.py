from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

from prompts import get_prompt_chat
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

os.environ["HF_TOKEN"] = "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"

SAMPLE_INPUT = "/data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data/commits_8/commit_data/django_django_commits.jsonl"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


with open(SAMPLE_INPUT, "r") as f:
    for i, line in enumerate(f):
        commit = json.loads(line)
        print(commit["message"])
        messages = get_prompt_chat('draft_from_diff', commit['files'][0]['current_content'],commit['files'][0]['patch'])
        break

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32000
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]




def get_llm_response(messages):
    # Convert messages to Qwen's chat format
    chat_text = ""
    for msg in messages:
        if msg["role"] == "system":
            chat_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            chat_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            chat_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    chat_text += "<|im_start|>assistant\n"  # Add the assistant prefix for generation

    # Initialize SGLang runtime
    runtime = RuntimeEndpoint(
        "Qwen/Qwen2.5-Coder-32B-Instruct"
    )

    # Generate response
    with runtime.init():
        response = gen(
            chat_text,
            temperature=0.1,
            top_p=0.9,
            max_tokens=120000,
            stop=["<|im_end|>"]
        )
    
    return response.text


# def load_model_and_tokenizer(model_id):
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="balanced",  # This will automatically balance across available GPUs
#         max_memory={0: "45GiB", 1: "45GiB"},
#         quantization_config=BitsAndBytesConfig(load_in_8bit=True)
#         # cache_dir=cache_dir
#     )
#     model = model.eval()
#     return tokenizer, model


# def generate_qa_pairs(model, tokenizer, input):
#     input_ids = tokenizer.apply_chat_template(
#         input,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     ).to(model.device)

#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=120000,
#         # attention_mask=input_ids["attention_mask"],
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.1,
#         top_p=0.9
#     )

#     response = outputs[0][input_ids.shape[-1]:]
#     return tokenizer.decode(response, skip_special_tokens=True)