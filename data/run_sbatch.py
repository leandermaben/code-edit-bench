import os
import subprocess

for i in range(1,11):
    script=f"""
    #!/bin/sh\n
    #SBATCH --partition=cpu\n
    #SBATCH --mem=300Gb\n
    #SBATCH --cpus-per-task=8\n
    #SBATCH -t 2-00:00:00              # time limit:  add - for days (D-HH:MM)\n
    #SBATCH --job-name=gen_commit_data_{i}\n
    #SBATCH --error=/home/lmaben/commit_data/job_outputs/%x__%j.err\n
    #SBATCH --output=/home/lmaben/commit_data/job_outputs/%x__%j.out\n
    #SBATCH --mail-type=ALL\n
    #SBATCH --mail-user=lmaben@andrew.cmu.edu\n
    
    source /data/tir/projects/tir7/user_data/lmaben/miniconda3/etc/profile.d/conda.sh\n
    conda activate code-edit-bench\n
    
    cd /home/lmaben/code-edit-bench\n
    
    python github-local-commit-extractor.py --repo_list data/split_repos/repo_list_{i} --output_dir /data/tir/projects/tir7/user_data/lmaben/code_edit_bench/data/commits_{i}/commit_data --repos_dir /data/tir/projects/tir7/user_data/lmaben/code_edit_bench/data/commits_{i}/repos --start_date 2022-01-01 --end_date 2024-10-01\n
    """

    script_filename = f"temp_job_script_{i}.sh"
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
