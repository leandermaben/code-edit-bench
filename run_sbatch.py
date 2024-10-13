import os
import subprocess

for i in range(1,11):
    script=f"""#!/bin/sh
#SBATCH --partition=general
#SBATCH --mem=80Gb
#SBATCH --cpus-per-task=8
#SBATCH -t 1-06:00:00              # time limit:  add - for days (D-HH:MM)
#SBATCH --job-name=gen_commit_data_{i}
#SBATCH --error=/home/lmaben/commit_data/job_outputs/%x__%j.err
#SBATCH --output=/home/lmaben/commit_data/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu    
source /data/tir/projects/tir7/user_data/lmaben/miniconda3/etc/profile.d/conda.sh
conda activate code-edit-bench
cd /home/lmaben/code-edit-bench
python github-local-commit-extractor.py --repo-list data/split_repos/repo_list_{i}.csv --output-dir /data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data/commits_{i}/commit_data --repos-dir /data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data/commits_{i}/repos --start-date 2022-01-01 --end-date 2024-10-01\n
"""

    script_filename = f"temp_job_script_{i}.sbatch"
    with open(script_filename, "w") as script_file:
        script_file.write(script)

    # Submit the job using sbatch
    try:
        result = subprocess.run(["sbatch", script_filename], check=True, capture_output=True, text=True)
        print(f"Job {i} submitted successfully. {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {i}: {e}")
    finally:
        # Clean up the temporary script file
        os.remove(script_filename)

    #Run the script
