import git
from git import Repo
from datetime import datetime
import json
import argparse
import subprocess
from typing import List, Dict, Any
from pathlib import Path
import logging
import shutil
import csv

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalGitExtractor:
    def __init__(self, output_dir: str, repos_dir: str):
        self.output_dir = Path(output_dir)
        self.repos_dir = Path(repos_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        #Check if metadata file exists
        metadata_file = self.output_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'repositories': {},
                'export_date': datetime.now().isoformat(),
                'total_commits': 0
            }

    def clone_repository(self, full_repo_name: str) -> Repo:
        repo_name = full_repo_name.split('/')[-1]
        local_path = self.repos_dir / repo_name
        
        if local_path.exists():
            logger.info(f"Repository already exists at {local_path}, pulling latest changes")
            repo = Repo(local_path)
            origin = repo.remotes.origin
            origin.pull()
            return repo
        
        logger.info(f"Cloning repository {full_repo_name}")
        return Repo.clone_from(f"https://github.com/{full_repo_name}.git", local_path)


    def get_commit_details(self, repo: Repo, commit) -> Dict[str, Any]:
        try:
            # Get the raw diff using git show
            git_show_command = [
                'git', 'show',
                '--format=format:',  # Suppress commit message header
                '--numstat',  # Get number statistics
                commit.hexsha
            ]

            # Run git show command for stats
            process = subprocess.Popen(
                git_show_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=repo.working_dir
            )
            show_output, _ = process.communicate()

            # Parse the numstat output
            stats = self.parse_numstat_output(show_output)

            # Get file contents and patches
            files = self.get_file_contents_and_patches(repo, commit)

            return {
                'sha': commit.hexsha,
                'author': f"{commit.author.name} <{commit.author.email}>",
                'date': commit.authored_datetime.isoformat(),
                'message': commit.message.strip(),
                **stats,
                'files': files
            }
        except Exception as e:
            logger.error(f"Error getting details for commit {commit.hexsha}: {e}")
            return None

    def parse_numstat_output(self, output: str) -> Dict[str, Any]:
        lines = output.strip().split('\n')
        files_changed = len(lines)
        total_additions = 0
        total_deletions = 0

        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    additions = int(parts[0]) if parts[0] != '-' else 0
                    deletions = int(parts[1]) if parts[1] != '-' else 0
                    total_additions += additions
                    total_deletions += deletions
                except ValueError:
                    # Skip lines that don't have valid number stats
                    continue

        return {
            'files_changed': files_changed,
            'additions': total_additions,
            'deletions': total_deletions,
        }

    def get_file_contents_and_patches(self, repo: Repo, commit) -> List[Dict[str, Any]]:
        files = []
        parent = commit.parents[0] if commit.parents else None

        # Get patches for all files in one go
        patch_command = ['git', 'show', '--format=format:', '--patch', commit.hexsha]
        process = subprocess.Popen(
            patch_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo.working_dir
        )
        patch_output, _ = process.communicate()

        # Split the patch output by file
        file_patches = self.split_patch_by_file(patch_output)

        for file in commit.stats.files:
            try:
                # Get current version
                current_content = self.get_file_content(repo, commit, file)

                # Get previous version
                previous_content = self.get_file_content(repo, parent, file) if parent else None

                file_info = {
                    'filename': file,
                    'status': 'modified',
                    'current_content': current_content,
                    'previous_content': previous_content,
                    'patch': file_patches.get(file, '')
                }

                if previous_content is None and current_content is not None:
                    file_info['status'] = 'added'
                elif previous_content is not None and current_content is None:
                    file_info['status'] = 'deleted'

                files.append(file_info)
            except Exception as e:
                logger.error(f"Error getting content for file {file} in commit {commit.hexsha}: {e}")

        return files

    def split_patch_by_file(self, patch_output: str) -> Dict[str, str]:
        file_patches = {}
        current_file = None
        current_patch = []

        for line in patch_output.split('\n'):
            if line.startswith('diff --git'):
                if current_file:
                    file_patches[current_file] = '\n'.join(current_patch)
                current_file = line.split()[-1][2:]  # Extract filename
                current_patch = [line]
            elif current_file:
                current_patch.append(line)

        if current_file:
            file_patches[current_file] = '\n'.join(current_patch)

        return file_patches

    def get_file_content(self, repo: Repo, commit, file_path: str) -> str:
        if commit is None:
            return None
        try:
            return repo.git.show(f'{commit.hexsha}:{file_path}')
        except git.exc.GitCommandError:
            # File doesn't exist in this commit
            return None

    def get_commits(self, repo: Repo, repo_name: str, start_date: str, end_date: str, lang=None, max_commits=-1) -> None:
        commits = []
        commit_count = 0
        file_count = 0
        
        # Convert dates to datetime objects
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        try:
            # Use git log with date filtering
            for commit in repo.iter_commits(
                    since=start.strftime('%Y-%m-%d'),
                    until=end.strftime('%Y-%m-%d')):
                if(commit.message.startswith('Merge pull request')):
                    logger.info(f"Skipping PR merge commit {commit.hexsha}")
                    continue
                try:
                    commit_data = self.get_commit_details(repo,commit)
                    commit_data['repo'] = repo_name
                    commits.append(commit_data)
                    commit_count += 1
                    file_count += len(commit_data['files'])
                    
                    if commit_count % 100 == 0:
                        logger.info(f"Processed {commit_count} commits for {repo_name}")

                    if commit_count % 500 == 0:
                        logger.info(f"Processed {commit_count} commits for {repo_name}")
                        self.save_commits_to_jsonl(commits, repo_name)
                        commits = []

                    if max_commits!=-1 and commit_count  == max_commits:
                        logger.info(f"Processed {commit_count} commits for {repo_name}.Stopping now.")
                        break
                except Exception as e:
                    logger.error(f"Error processing commit {commit.hexsha}: {e}")
        except Exception as e:
            logger.error(f"Error iterating commits for {repo_name}: {e}")

        if len(commits) > 0:
            self.save_commits_to_jsonl(commits, repo_name)
        
        # Update metadata
        self.metadata['repositories'][repo_name] = {
            'commit_count': commit_count,
            'start_date': start_date,
            'end_date': end_date,
            'language': lang,
            'file_count':file_count
        }
        self.metadata['total_commits'] += commit_count
        return commit_count


    def save_commits_to_jsonl(self, commits: List[Dict[str, Any]], repo_name: str):
        if not commits:
            logger.warning(f"No commits to save for {repo_name}")
            return
        
        output_file = self.output_dir / f"{repo_name.replace('/', '_')}_commits.jsonl"
        
        with open(output_file, 'a', encoding='utf-8') as jsonl_file:
            for commit in commits:
                jsonl_file.write(json.dumps(commit) + '\n')

    def save_metadata(self):
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def cleanup(self, full_repo_name: str=None):
        if full_repo_name:
            repo_name = full_repo_name.split('/')[-1]
            local_path = self.repos_dir / repo_name
            logger.info(f"Cleaning up repositories directory: {local_path}")
            if local_path.exists():
                shutil.rmtree(local_path,ignore_errors=True)
        else:
            logger.info(f"Cleaning up repositories directory: {self.repos_dir}")
            if self.repos_dir.exists():
                shutil.rmtree(self.repos_dir,ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Extract Git commit information")
    parser.add_argument("--output-dir", type=str, default="data/commits/git_commits_data",
                        help="Directory to store output files (default: data/commits/git_commits_data)")
    parser.add_argument("--repos-dir", type=str, default="data/commits/git_repos",
                        help="Directory to clone repositories (default: data/commits/git_repos)")
    parser.add_argument("--repo-list", type=str, default="data/repo_list.csv",
                        help="CSV file containing list of repositories to analyze")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                        help="Start date for commit extraction (default: 2020-01-01)")
    parser.add_argument("--end-date", type=str, default="2024-03-31",
                        help="End date for commit extraction (default: 2024-03-31)")
    parser.add_argument("--max-commit-per-repo", type=int, default="-1",
                        help="Max number of commits to extract per repository (default: -1). -1 -> No limit.")
    args = parser.parse_args()


    # List of repositories to analyze
    with open(args.repo_list, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        repositories = [(row['full_name'],row['language']) for row in reader]

    extractor = LocalGitExtractor(args.output_dir, args.repos_dir)

    try:
        for full_repo_name, language in repositories:
            logger.info(f"Processing repository: {full_repo_name}")
            try:
                if full_repo_name in extractor.metadata['repositories']:
                    logger.info(f"Repository {full_repo_name} already processed. Skipping.")
                    continue
                repo = extractor.clone_repository(full_repo_name)
                commit_count = extractor.get_commits(repo, full_repo_name, args.start_date, args.end_date, language, args.max_commit_per_repo)
                logger.info(f"Found and saved {commit_count} commits for {full_repo_name}")
                extractor.save_metadata()
                logger.info(f"Saved metadata to {args.output_dir}/metadata.json")
            except Exception as e:
                logger.error(f"Failed to process repository {full_repo_name}: {e}")
            finally:
                # Cleanup repository
                extractor.cleanup(full_repo_name)
        extractor.save_metadata()
        logger.info(f"Saved metadata to {args.output_dir}/metadata.json")
    finally:
        # Cleanup repositories to save disk space
        extractor.cleanup()

if __name__ == "__main__":
    main()
