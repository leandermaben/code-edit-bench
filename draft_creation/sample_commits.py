import json
import numpy as np
import tiktoken
from generate_draft import get_draft_from_specs, get_draft_from_diff
from draft_utils import count_tokens, wait_for_server
import logging
from tqdm import tqdm
import os

DATA_BASE_PATH = '/data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data'
MAX_FILES_PER_REPO = 20 #12 HERE
MAX_FILES_PER_COMMIT = 4 #3 HERE
MAX_REPOS_PER_SPLIT = 100 # 70 HERE
SAMPLED_REPOS_PER_SPLIT = 110
MAX_PATCH_TOKENS = 16000
MAX_ORIGINAL_FILE_TOKENS = 16000
MIN_PATCH_TOKENS = 100

MODEL = 'Qwen/Qwen2.5-72B-Instruct'

wait_for_server(MODEL)

for split in [2,3,5]: #11 #HERE
    commits_filtered = set()
    repos_filtered = set()
    sampled_commits_file_name = f'{DATA_BASE_PATH}/sampled_commits_{MODEL.replace("/","_")}_split_{split}.jsonl'
    sampled_repos_meta_file_name = f'data/sampled_data/sampled_repos_{MODEL.replace("/","_")}_split_{split}.txt'
    sampled_commits_meta_file_name = f'data/sampled_data/sampled_commits_{MODEL.replace("/","_")}_split_{split}.txt'
    repos_processed = 0
    with open(f'{DATA_BASE_PATH}/commits_{split}/commit_data/metadata.json', 'r') as f:
        repos = json.load(f)['repositories']
    sampled_repos = np.random.choice(list(repos.keys()), SAMPLED_REPOS_PER_SPLIT, replace=False)
    logging.info(f'split {split} sampled {len(sampled_repos)} repos out of {len(repos)}')
    repos_filtered.update(sampled_repos)
    for repo in tqdm(sampled_repos, desc=f'Processing repos in split {split}'):
        if repos_processed >= MAX_REPOS_PER_SPLIT:
            break
        repo_file_counter = 0
        num_commits = repos[repo]['commit_count']
        sampled_commits = np.random.choice(np.arange(num_commits), min(num_commits,10), replace=False)
        if not os.path.exists(f'{DATA_BASE_PATH}/commits_{split}/commit_data/{repo.replace("/","_")}_commits.jsonl'):
            logging.warning(f'{repo} not found')
            continue
        with open(f'{DATA_BASE_PATH}/commits_{split}/commit_data/{repo.replace("/","_")}_commits.jsonl', 'r') as commits_file:
            for i,line in enumerate(tqdm(commits_file, desc=f'Processing commits in {repo}', leave=False)):
                if repo_file_counter >= MAX_FILES_PER_REPO:
                    break
                if i not in sampled_commits:
                    continue
                commit_data = json.loads(line)
                np.random.shuffle(commit_data['files'])
                files = commit_data['files']
                sampled_file_data=[]
                for j,file in enumerate(files):
                    if len(sampled_file_data) >= MAX_FILES_PER_COMMIT:
                        break
                    try:
                        patch_token_count = count_tokens(file['patch'])
                        original_file_token_count = count_tokens(file['previous_content'])
                        if patch_token_count > MIN_PATCH_TOKENS and patch_token_count < MAX_PATCH_TOKENS and original_file_token_count < MAX_ORIGINAL_FILE_TOKENS:
                            # detailed_specs, draft_from_detailed_specs = get_draft_from_specs(file['previous_content'], file['patch'], detailed_spec=True) #ADD model
                            # higher_level_specs, draft_from_higher_level_specs = get_draft_from_specs(file['previous_content'], file['patch'], detailed_spec=False) #ADD model
                            draft_from_diff = get_draft_from_diff(file['previous_content'], file['patch'],model=MODEL) #ADD model

                            #file['detailed_specs'] = detailed_specs
                            #file['draft_from_detailed_specs'] = draft_from_detailed_specs
                            # file['higher_level_specs'] = higher_level_specs
                            # file['draft_from_higher_level_specs'] = draft_from_higher_level_specs
                            file['draft_from_diff'] = draft_from_diff

                            sampled_file_data.append(file)
                            commits_filtered.add((commit_data['sha'],file['filename']))
                            repo_file_counter += 1
                    except Exception as e:
                        logging.warning(f'Error processing file {file["filename"]} in commit {commit_data["sha"]}: {e}')
                if len(sampled_file_data) == 0:
                    continue
                filtered_commit_data = {
                    'sha': commit_data['sha'],
                    'generation_model': MODEL,
                    'split': split,
                    'repo': repo,
                    'language': repos[repo]['language'],
                    'message': commit_data['message'],
                    'files_changed': len(sampled_file_data),
                    'files': sampled_file_data
                }
    
                with open(sampled_commits_file_name, 'a') as sampled_commits_file:
                    sampled_commits_file.write(json.dumps(filtered_commit_data) + '\n')
    
                
    with open(sampled_commits_meta_file_name, 'w') as f:
        for commit in commits_filtered:
            f.write(f'{commit[0]}:{commit[1]}\n')

    with open(sampled_repos_meta_file_name, 'w') as f:
        for repo in repos_filtered:
            f.write(f'{repo}\n')
            logging.info(f'Processed {repo}')
            repos_processed += 1

    logging.info(f'Processed {len(commits_filtered)} commits in split {split}')
                



#1369 repos in total







