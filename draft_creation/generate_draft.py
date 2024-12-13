import os
from prompts import get_prompt_chat
import json
from openai import OpenAI

MODEL = 'Qwen/Qwen2.5-Coder-32B-Instruct'

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_draft_from_diff(original_file_content, patch, model=MODEL):
    messages = get_prompt_chat('draft_from_diff', original_file_content, patch)
    draft = client.chat.completions.create(
            model=model,
            messages=messages
        )
    return draft.choices[0].message.content

def get_draft_from_specs(original_file_content, patch,detailed_spec=True):
    if detailed_spec:
        specs_prompt = get_prompt_chat('detailed_specs', original_file_content, patch)
    else:
        specs_prompt = get_prompt_chat('higher_level_specs', original_file_content, patch)
    specs = client.chat.completions.create(
            model=MODEL,
            messages=specs_prompt
        )
    draft_prompt = get_prompt_chat('draft_from_specs', original_code=original_file_content, specs=specs.choices[0].message.content.replace('<specs>', '').replace('</specs>', ''))
    draft = client.chat.completions.create(
            model=MODEL,
            messages=draft_prompt
        )
    return specs.choices[0].message.content, draft.choices[0].message.content

if __name__ == "__main__":
    SAMPLE_INPUT = "/data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data/commits_8/commit_data/django_django_commits.jsonl"

    with open(SAMPLE_INPUT, "r") as f:
        data = []
        for i, line in enumerate(f):
            if i>5:
                break
            commit = json.loads(line)
            print(commit["message"])

            detailed_specs, draft_from_detailed_specs = get_draft_from_specs(commit['files'][0]['previous_content'], commit['files'][0]['patch'], detailed_spec=True)
            higher_level_specs, draft_from_higher_level_specs = get_draft_from_specs(commit['files'][0]['previous_content'], commit['files'][0]['patch'], detailed_spec=False)
            draft_from_diff = get_draft_from_diff(commit['files'][0]['previous_content'], commit['files'][0]['patch'])
            data.append({
                "original_code": commit['files'][0]['previous_content'],
                "patch": commit['files'][0]['patch'],
                "new_code": commit['files'][0]['current_content'],
                "detailed_specs": detailed_specs,
                "draft_from_detailed_specs": draft_from_detailed_specs,
                "higher_level_specs": higher_level_specs,
                "draft_from_higher_level_specs": draft_from_higher_level_specs,
                "draft_from_diff": draft_from_diff
            })
        with open('data/draft_samples.jsonl', 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

