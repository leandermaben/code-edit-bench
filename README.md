# Code Edit Bench

This repository contains code to generate a dataset of commits from top rated (based on number of stars) repositories on Git that are based on 14 different languages. This will serve as a benchmark for code editing tasks.

## Directory Structure
**Data** <br> 
&nbsp;&nbsp;&nbsp;&nbsp;**repo_list** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Contains information from top rated repositories in csv files (One csv file per main language) extracted using https://seart-ghs.si.usi.ch/* <br>
&nbsp;&nbsp;&nbsp;&nbsp;**repo_list.csv** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Consolidated list of repo names, language and stars of the top 100 repositories from each csv under the repo_list directory* <br>

**github-commit-extractor.py** <br>
*Python script to extract commits using gitHub API (Not used due to API rate issues)* <br>
**github-local-commit-extractor.py** <br>
*Python script to extract commits using local git repositories (using the list of repos in data/repo_list.csv)*<br>
**prepare_repo_list.py** <br>
*Used to generate data/repo_list.csv from data/repo_list/\*.csv* <br> 
**statistics.py** <br>
*Can be used to generate statistics on the dataset from the jsonl files generated after running github-local-commit-extractor.py*
