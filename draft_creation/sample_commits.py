import json
import numpy as np
import tiktoken
from generate_draft import get_draft_from_specs, get_draft_from_diff
from draft_utils import count_tokens
import logging

DATA_BASE_PATH = '/data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data'
SAMPLED_COMMITS_FILE = f'{DATA_BASE_PATH}/sampled_commits.jsonl'
MAX_FILES_PER_REPO = 15
MAX_FILES_PER_COMMIT = 3
REPOS_PER_SPLIT = 70

repositories = {}
commits_filtered = set()
repos_filtered = set()

for split in range(1,11):
    with open(f'{DATA_BASE_PATH}/commits_{split}/commit_data/metadata.json', 'r') as f:
        repos = json.load(f)['repositories']
    sampled_repos = np.random.choice(list(repos.keys()), REPOS_PER_SPLIT, replace=False)
    logging.info(f'split {split} sampled {len(sampled_repos)} repos out of {len(repos)}')
    repos_filtered.update(sampled_repos)
    for repo in sampled_repos:
        repo_file_counter = 0
        num_commits = repos[repo]['commit_count']
        sampled_commits = np.random.choice(np.arange(num_commits), min(num_commits,10), replace=False)
        with open(f'{DATA_BASE_PATH}/commits_{split}/commit_data/{repo.replace("/","_")}_commits.jsonl', 'r') as commits_file:
            for i,line in enumerate(commits_file):
                if repo_file_counter >= MAX_FILES_PER_REPO:
                    break
                if i not in sampled_commits:
                    continue
                commit_data = json.loads(line)
                files = np.random.shuffle(commit_data['files'])
                sampled_file_data=[]
                for j,file in enumerate(files):
                    if len(sampled_file_data) >= MAX_FILES_PER_COMMIT:
                        break
                    patch_token_count = count_tokens(file['patch'])
                    original_file_token_count = count_tokens(file['previous_content'])
                    if patch_token_count > 100 and patch_token_count < 60000 and original_file_token_count < 60000:
                        detailed_specs, draft_from_detailed_specs = get_draft_from_specs(file['previous_content'], file['patch'], detailed_spec=True)
                        higher_level_specs, draft_from_higher_level_specs = get_draft_from_specs(file['previous_content'], file['patch'], detailed_spec=False)
                        draft_from_diff = get_draft_from_diff(file['previous_content'], file['patch'])

                        file['detailed_specs'] = detailed_specs
                        file['draft_from_detailed_specs'] = draft_from_detailed_specs
                        file['higher_level_specs'] = higher_level_specs
                        file['draft_from_higher_level_specs'] = draft_from_higher_level_specs
                        file['draft_from_diff'] = draft_from_diff

                        sampled_file_data.append(file)
                        commits_filtered.add((commit_data['sha'],file['filename']))
                        repo_file_counter += 1
                filtered_commit_data = {
                    'sha': commit_data['sha'],
                    'split': split,
                    'repo': repo,
                    'message': commit_data['message'],
                    'files_changed': len(sampled_file_data),
                    'files': sampled_file_data
                }
    
                with open(SAMPLED_COMMITS_FILE, 'a') as sampled_commits_file:
                    sampled_commits_file.write(json.dumps(filtered_commit_data) + '\n')
        logging.info(f'Processed {repo}')
                
logging.info(f'Processed {len(commits_filtered)} commits')

with open(f'data/sampled_commits.txt', 'w') as f:
    for commit in commits_filtered:
        f.write(f'{commit[0]}:{commit[1]}\n')

with open(f'data/sampled_repos.txt', 'w') as f:
    for repo in repos_filtered:
        f.write(f'{repo}\n')
                    

                

            
            

        #repositories.update(json.load(f)['repositories'])



#1369 repos in total







