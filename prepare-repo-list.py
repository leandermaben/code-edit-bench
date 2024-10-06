# Read all csv files in data/repo_lists and write list of repository names, language and stars to a csv file

import csv
from pathlib import Path


def prepare_repo_list(repo_list_dir: Path, output_file: Path) -> None:
    all_repos = []
    for file in repo_list_dir.glob('*.csv'):
        with open(file, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            file_repos = []
            for row in reader:
                try:
                    stars = int(row['stargazers'])
                    file_repos.append((row['name'], row['mainLanguage'], stars))
                except (KeyError, ValueError):
                    print(f"Warning: Skipping row due to missing or invalid data in {file}")

            # Sort by number of stars (descending) and pick top 100
            file_repos.sort(key=lambda x: x[2], reverse=True)
            all_repos.extend(file_repos[:100])

    # Remove duplicates while preserving order
    unique_repos = list(dict.fromkeys(all_repos))

    with open(output_file, 'w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['full_name', 'language', 'stars'])  # Write header
        for repo, lang, stars in unique_repos:
            writer.writerow([repo, lang, stars])


if __name__ == '__main__':
    prepare_repo_list(Path('data/repo_lists'), Path('data/repo_list.csv'))

