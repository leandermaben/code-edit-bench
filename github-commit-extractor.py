"""

This script is not used because of API rate issues with GitHub API.

"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import json
import os
from typing import List, Dict, Any
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubCommitExtractor:
    def __init__(self, token: str, output_dir: str):
        self.token = token
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            'repositories': {},
            'export_date': datetime.now().isoformat(),
            'total_commits': 0
        }
        retry_strategy = Retry(
            total=8,  # number of retries
            backoff_factor=2,  # wait 1, 2, 4, 8, 16, 32, 64, 128 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        )

        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def make_request(self, url: str, params: Dict[str, Any] = None) -> requests.Response:
        while True:
            try:
                response = self.session.get(url, headers=self.headers, params=params)

                if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    wait_time = max(reset_time - time.time(), 0)
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time + 1)  # Add 1 second buffer
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.RetryError):
                    logger.error(f"Max retries exceeded for {url}")
                    raise
                logger.error(f"Error making request to {url}: {e}")
                raise

    def get_commits(self, repo: str, start_date: str, end_date: str, lang: str = None) -> List[Dict[str, Any]]:
        commits = []
        page = 1
        repo_commit_count = 0
        
        while True:
            url = f'{self.base_url}/repos/{repo}/commits'
            params = {
                'since': f'{start_date}T00:00:00Z',
                'until': f'{end_date}T23:59:59Z',
                'page': page,
                'per_page': 100
            }
            
            response = self.make_request(url, params)
            
            if response.status_code == 404:
                print(f"Repository {repo} not found")
                return []
            
            response.raise_for_status()
            page_commits = response.json()
            
            if not page_commits:
                break
                
            for commit in page_commits:
                commit_data = self.get_commit_details(repo, commit['sha'])
                commits.append(commit_data)
                repo_commit_count += 1
                
            page += 1
        
        # Update metadata for this repository
        self.metadata['repositories'][repo] = {
            'commit_count': repo_commit_count,
            'start_date': start_date,
            'end_date': end_date,
            'lang': lang
        }
        self.metadata['total_commits'] += repo_commit_count
        
        return commits

    def get_commit_details(self, repo: str, commit_sha: str) -> Dict[str, Any]:
        url = f'{self.base_url}/repos/{repo}/commits/{commit_sha}'
        response = self.make_request(url)
        response.raise_for_status()
        
        commit_data = response.json()
        return {
            'repo': repo,
            'sha': commit_data['sha'],
            'author': commit_data['commit']['author']['name'],
            'date': commit_data['commit']['author']['date'],
            'message': commit_data['commit']['message'],
            'files_changed': len(commit_data['files']),
            'additions': sum(f['additions'] for f in commit_data['files']),
            'deletions': sum(f['deletions'] for f in commit_data['files']),
            'files': [{
                'filename': f['filename'],
                'status': f['status'],
                'additions': f['additions'],
                'deletions': f['deletions'],
                'patch': f.get('patch', '')
            } for f in commit_data['files']]
        }

    def save_commits_to_jsonl(self, commits: List[Dict[str, Any]], repo: str):
        if not commits:
            print(f"No commits to save for {repo}")
            return
        
        output_file = self.output_dir / f"{repo.replace('/', '_')}_commits.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for commit in commits:
                jsonl_file.write(json.dumps(commit) + '\n')

    def save_metadata(self):
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

def main():
    # Replace with your GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        raise ValueError("GitHub token not found. Set the GITHUB_TOKEN environment variable.")

    # Output directory
    output_dir = 'github_commits_data'

    # List of repositories to analyze
    repositories = [
        'torvalds/linux'
    ]

    # Date range
    start_date = '2020-01-01'
    end_date = '2024-03-31'

    extractor = GitHubCommitExtractor(github_token, output_dir)

    for repo in repositories:
        print(f"Extracting commits from {repo}")
        try:
            commits = extractor.get_commits(repo, start_date, end_date, 'C')
            extractor.save_commits_to_jsonl(commits, repo)
            print(f"Found and saved {len(commits)} commits for {repo}")
        except requests.exceptions.RequestException as e:
            print(f"Error extracting commits from {repo}: {e}")
    
    # Save metadata
    extractor.save_metadata()
    print(f"Saved metadata to {output_dir}/metadata.json")

if __name__ == "__main__":
    main()
