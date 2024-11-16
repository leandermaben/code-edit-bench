import json
import tiktoken
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate statistics for JSONL files")
    parser.add_argument("--input_dir", type=str, default="data/commits/git_commit_data", help="Directory containing JSONL files")
    parser.add_argument("--output", type=str, default="data/commits/stats_output.json", help="Output JSON file name")
    return parser.parse_args()


def clean_text(text):
    # Remove or replace surrogate characters
    return text.encode('utf-8', 'ignore').decode('utf-8')

def count_tokens(text, max_chunk_size=10000):
    encoding = tiktoken.get_encoding("o200k_base")
    total_tokens = 0
    
    # Process text in chunks
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        try:
            tokens = encoding.encode(clean_text(chunk), disallowed_special=())
            total_tokens += len(tokens)
        except Exception as e:
            print(f"Warning: Error processing chunk {i}: {e}")
            # Optional: use a simpler fallback method
            total_tokens += len(chunk.split())  # rough approximation
            
    return total_tokens

# def count_tokens(text: str) -> int:
#     encoding = tiktoken.get_encoding("o200k_base")
#     try:
#         return len(encoding.encode(clean_text(text), disallowed_special=()))
#     except Exception as e:
#         print(f"Error encoding text: {e}")
#         return len(text.split())


def count_lines(text: str) -> int:
    return len(text.splitlines())


def calculate_stats(values: List[int]) -> Dict[str, float]:
    return {
        "total": float(sum(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "25%": float(np.percentile(values, 25)),
        "50%": float(np.percentile(values, 50)),
        "75%": float(np.percentile(values, 75)),
        "max": float(np.max(values))
    }


def process_jsonl_file(file_path: Path) -> pd.DataFrame:
    data = []
    lines_processed = 0
    with file_path.open('r') as f:
        for line in f:
            commit = json.loads(line)
            for file in commit['files']:
                current_content = file.get('current_content', '')
                previous_content = file.get('previous_content', '')
                patch = file.get('patch', '')

                data.append({
                    'current_tokens': count_tokens(current_content) if current_content else 0,
                    'current_lines': count_lines(current_content) if current_content else 0,
                    'previous_tokens': count_tokens(previous_content) if previous_content else 0,
                    'previous_lines': count_lines(previous_content) if previous_content else 0,
                    'patch_tokens': count_tokens(patch) if patch else 0,
                    'patch_lines': count_lines(patch) if patch else 0
                })
            lines_processed+=1
            if lines_processed % 100 ==0:
                gc.collect()

    return pd.DataFrame(data)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = args.output

    all_data = []

    for jsonl_file in input_dir.glob('*.jsonl'):
        print(f"Processing {jsonl_file}")
        df = process_jsonl_file(jsonl_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    
    for data_item in all_data:
        del data_item
        
    gc.collect()

    stats = {
        'current_content': {
            'tokens': calculate_stats(combined_df['current_tokens']),
            'lines': calculate_stats(combined_df['current_lines'])
        },
        'previous_content': {
            'tokens': calculate_stats(combined_df['previous_tokens']),
            'lines': calculate_stats(combined_df['previous_lines'])
        },
        'patch': {
            'tokens': calculate_stats(combined_df['patch_tokens']),
            'lines': calculate_stats(combined_df['patch_lines'])
        }
    }

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    main()