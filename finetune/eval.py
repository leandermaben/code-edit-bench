import argparse
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import GenerationConfig
from Levenshtein import distance
from train_utils import extract_code_from_tags, get_messages
from unsloth.chat_templates import get_chat_template
import torch
import numpy as np
from tqdm import tqdm
import os
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the fine-tuned model')
    parser.add_argument('--test_data_path', type=str, default='data/finetune/test.jsonl',
                      help='Path to test data JSONL file')
    parser.add_argument('--max_seq_length', type=int, default=131072,
                      help='Maximum sequence length')
    parser.add_argument('--chat_template', type=str, default='qwen-2.5',
                      help='Chat template to use')
    parser.add_argument('--out_dir',type=str,default='data/eval_results')
    parser.add_argument('--num_test_samples',type=int,default='200')
    return parser.parse_args()



def main():
    stats ={
        'exact_match':[],
        'edit_distance':[],
        'exact_match_lang':{},
        'edit_distance_lang':{}
    }
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto detection
    )
    model = FastLanguageModel.for_inference(model)
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=args.chat_template,  # Use appropriate template for your model
    )
    
    model.eval()
    
    # Load test data
    with open(args.test_data_path) as f:
        test_data = [json.loads(line) for line in f] #TODO: Remove this
    np.random.seed(42)
    test_data = np.random.choice(test_data,args.num_test_samples)
    test_data = [{'previous_content': d['file_previous_content'],
                  'current_content': d['file_current_content'],
                  'draft': d['file_draft_from_diff'],'sha':d['sha'],'language':d['language']} for d in test_data]
    
    # Prepare generation config
    #TODO: Check 
    gen_config = GenerationConfig(
        max_new_tokens=10000,
        do_sample=False,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    
    for item in tqdm(test_data):
        # Create conversation format
        messages = get_messages(item['previous_content'], item['draft'], item['current_content'])
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False)  # Exclude the last message (ground truth)
        ground_truth = messages[-1]['content']

        print(f"Prompt: {prompt}")
        
        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
            )
        
        print(f"Output Length: {len(outputs)}")
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove any template markers and extract from tags
        prediction = prediction.replace(prompt, "").strip()
        print(f"Prediction: {prediction}")
        print(f"Ground Truth: {ground_truth}")
        prediction = extract_code_from_tags(prediction, tag="updated_code")
        
        # Ground truth should also be extracted from tags if present
        ground_truth = extract_code_from_tags(ground_truth, tag="updated_code")
   
        
        # Compute metrics
        exact_match = prediction == ground_truth
        edit_dist = distance(prediction, ground_truth)
        
        stats['exact_match'].append(exact_match)
        stats['edit_distance'].append(edit_dist)
        if not item['language'] in stats['exact_match_lang']:
            stats['exact_match_lang'][item['language']] = []
            stats['edit_distance_lang'][item['language']] = []
        stats['exact_match_lang'][item['language']].append(exact_match)
        stats['edit_distance_lang'][item['language']].append(edit_dist)
    
    # Compute final metrics
    stats['exact_match_score'] = np.mean(stats['exact_match'])
    stats['avg_edit_distance'] = np.mean(stats['edit_distance'])
    for lang in stats['exact_match_lang']:
        stats['exact_match_lang'][lang] = np.mean(stats['exact_match_lang'][lang])
        stats['edit_distance_lang'][lang] = np.mean(stats['edit_distance_lang'][lang])
    
    print(f"Exact Match Score: {stats['exact_match_score']:.4f}")
    print(f"Average Edit Distance: {stats['avg_edit_distance']:.4f}")
    with open(os.path.join(args.out_dir,f'{args.model_path.split("/")[-2]}.json'),'w') as f:
        json.dump(stats,f,indent=4)

if __name__ == "__main__":
    main()
