import csv
import math
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Split a repository list CSV into multiple files")
    parser.add_argument("--input_file", type=str, default="data/repo_list.csv", help="Path to the input CSV file containing the list of repositories")
    parser.add_argument("--output_dir", type=str, default="data/split_repos", help="Directory to save the output files (default: current directory)")
    parser.add_argument("--num_files", type=int, default=10, help="Number of files to split into (default: 10)")
    return parser.parse_args()

def split_list(input_list, num_parts):
    avg = len(input_list) / float(num_parts)
    out = []
    last = 0.0
    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg
    return out

def main():
    args = parse_arguments()
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    num_files = args.num_files

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the input CSV file
    repositories = []
    with input_file.open('r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'full_name' in row and row['full_name'].strip():
                repositories.append([row['full_name'].strip(), row['language'], row['stars']])

    # Split the list of repositories
    split_repos = split_list(repositories, num_files)

    # Write to output files
    for i, repo_list in enumerate(split_repos, 1):
        output_file = output_dir / f"repo_list_{i}.csv"
        with output_file.open('w') as f:
            f.write("full_name,language,stars\n")
            for full_name, language, stars in repo_list:
                f.write(f"{full_name},{language},{stars}\n")
        print(f"Created {output_file} with {len(repo_list)} repositories")

    print(f"Split {len(repositories)} repositories into {num_files} files")

if __name__ == "__main__":
    main()