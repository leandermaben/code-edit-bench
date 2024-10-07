import json
import tiktoken
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate statistics for JSONL files")
    parser.add_argument("--input_dir", type=str, default="data/commits/git_commit_data", help="Directory containing JSONL files")
    parser.add_argument("--output", type=str, default="data/commits/stats_output.json", help="Output JSON file name")
    return parser.parse_args()


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def count_lines(text: str) -> int:
    return len(text.splitlines())


def calculate_stats(values: List[int]) -> Dict[str, float]:
    return {
        "total": sum(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "25%": np.percentile(values, 25),
        "50%": np.percentile(values, 50),
        "75%": np.percentile(values, 75),
        "max": np.max(values)
    }


def process_jsonl_file(file_path: Path) -> pd.DataFrame:
    data = []
    with file_path.open('r') as f:
        for line in f:
            commit = json.loads(line)
            for file in commit['files']:
                current_content = file.get('current_content', '')
                previous_content = file.get('previous_content', '')
                patch = file.get('patch', '')

                data.append({
                    'current_tokens': count_tokens(current_content),
                    'current_lines': count_lines(current_content),
                    'previous_tokens': count_tokens(previous_content),
                    'previous_lines': count_lines(previous_content),
                    'patch_tokens': count_tokens(patch),
                    'patch_lines': count_lines(patch)
                })

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