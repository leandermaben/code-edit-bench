import json
import argparse
import numpy as np
import re
import os
from Levenshtein import distance
from openai import OpenAI
import time
from tqdm import tqdm
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def wait_for_server(model):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Loop until the server is up
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'Hello'"}],
                max_tokens=10
            )
            print("Server is up and running!")
            print("Response:", response.choices[0].message.content)
            break  # Exit the loop if the server is up
        except Exception as e:
            print("Server is not responding, retrying in 5 seconds...")
            time.sleep(20)  # Wait for 5 seconds before retrying


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned model')
    parser.add_argument('--train_data_path', type=str, default='data/finetune/train.jsonl',
                      help='Path to train data JSONL file')
    parser.add_argument('--test_data_path', type=str, default='data/finetune/test.jsonl',
                      help='Path to test data JSONL file')
    parser.add_argument('--few_shot_examples', type=int, default=0)
    parser.add_argument('--num_test_examples', type=int, default=10)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Coder-3B-Instruct',
                      help='Model Name')
    return parser.parse_args()

args = parse_args()

wait_for_server(args.model)

with open(args.test_data_path) as f:
    test_data = [json.loads(line) for line in f]
np.random.seed(42)
test_data = np.random.choice(test_data,args.num_test_examples)

with open(args.train_data_path) as f:
    train_data = [json.loads(line) for line in f]


few_shot_prompt_to_apply_diff = {
    'system_prompt':"""
        Code changes will be provided in the form of a draft. You will need to apply the draft to the original code. 
        The original code will be enclosed within `<original_code>` tags.
        The draft will be enclosed within `<update_snippet>` tags.
        You need to output the update code within `<updated_code>` tags.

        Within the `<updated_code>` include only the final code after updation. Do not include any explanations or other content within these tags.

        Some examples are provided below within the `<examples>` tags.

        <examples>
        {examples}
        </examples>

    """,
    'user_prompt':"""

        Apply the update snippet to the original code.
    
        <original_code>
        {original_code}
        </original_code>

        <update_snippet>
        {update_snippet}
        </update_snippet>
    """
}

stats ={
    'exact_match':[],
    'edit_distance':[],
    'exact_match_lang':{},
    'edit_distance_lang':{}
}

def extract_code_from_tags(text, tag):
    import re
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

for item in tqdm(test_data):
    few_shot_examples = np.random.choice(train_data,args.few_shot_examples)
    few_shot_examples_str = "\n".join([f"Example {i+1}:\n<original_code>\n{example['file_previous_content']}\n</original_code>\n <update_snippet>\n{example['file_patch']}\n</update_snippet>\n<updated_code>\n{example['file_current_content']}\n</updated_code>" for i, example in enumerate(few_shot_examples)])
    few_shot_prompt = few_shot_prompt_to_apply_diff['system_prompt'].format(examples=few_shot_examples_str)

    #print(few_shot_prompt)

    messages = [
        {'role':'system', 'content':few_shot_prompt},
        {'role':'user', 'content':few_shot_prompt_to_apply_diff['user_prompt'].format(original_code=item['file_previous_content'], update_snippet=item['file_patch'])}
    ]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages
    )

    generated_code = response.choices[0].message.content
    #print(f"Generated Code: {generated_code}")
    generated_code = extract_code_from_tags(generated_code, tag="updated_code")
    ground_truth = item['file_current_content']
    exact_match = generated_code == ground_truth
    edit_dist = distance(generated_code, ground_truth)
        
    stats['exact_match'].append(exact_match)
    stats['edit_distance'].append(edit_dist)
    if not item['language'] in stats['exact_match_lang']:
        stats['exact_match_lang'][item['language']] = []
        stats['edit_distance_lang'][item['language']] = []
    stats['exact_match_lang'][item['language']].append(exact_match)
    stats['edit_distance_lang'][item['language']].append(edit_dist)
    
stats['exact_match_score'] = np.mean(stats['exact_match'])
stats['avg_edit_distance'] = np.mean(stats['edit_distance'])
for lang in stats['exact_match_lang']:
    stats['exact_match_lang'][lang] = np.mean(stats['exact_match_lang'][lang])
    stats['edit_distance_lang'][lang] = np.mean(stats['edit_distance_lang'][lang])

print(f"Exact Match Score: {stats['exact_match_score']:.4f}")
print(f"Average Edit Distance: {stats['avg_edit_distance']:.4f}")
with open(os.path.join('data/in_context',f'{args.model.split("/")[-1]}_{args.few_shot_examples}.json'),'w') as f:
    json.dump(stats,f,indent=4)


 




