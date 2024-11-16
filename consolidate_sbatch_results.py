import json
import shutil
from pathlib import Path
import argparse


def consolidate_results(input_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_metadata = {
        'repositories': {},
        'total_commits': 0
    }

    for input_dir in input_dirs:
        input_path = Path(input_dir)

        # Process metadata
        metadata_file = input_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                consolidated_metadata['repositories'].update(metadata['repositories'])
                consolidated_metadata['total_commits'] += metadata['total_commits']

        # Move JSONL files
        for jsonl_file in input_path.glob('*.jsonl'):
            new_path = output_dir / jsonl_file.name
            if new_path.exists():
                # Handle filename conflicts
                print('WARN: ***Duplicate filename detected***')
                i = 1
                while new_path.exists():
                    new_path = output_dir / f"{jsonl_file.stem}_{i}{jsonl_file.suffix}"
                    i += 1
            shutil.move(str(jsonl_file), str(new_path))

    # Save consolidated metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(consolidated_metadata, f, indent=2)

    print(f"Results consolidated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate SBATCH job results")
    parser.add_argument('--input_dirs', nargs='+', help='Input directories containing job results')
    parser.add_argument('--output-dir', default='consolidated_results',
                        help='Output directory for consolidated results')
    args = parser.parse_args()

    consolidate_results(args.input_dirs, args.output_dir)


if __name__ == "__main__":
    main()