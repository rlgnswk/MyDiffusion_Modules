#!/bin/bash
#SBATCH --job-name=run_models          # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 task per node)
#SBATCH --gres=gpu:1                   # Number of GPUs per node
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --time=0-12:00:00               # Maximum time limit for the job
#SBATCH --partition=P2                 # Use the correct partition (e.g., P2)
#SBATCH --mem=100G
#SBATCH --output=ddim_inversion_hf.log # Output log file

# Load necessary modules (if needed)
# module load python/3.8
# module load cuda/11.8

# Run the Python script
source ${HOME}/.bashrc
source /shared/s2/lab01/gihoon/anaconda3/bin/activate
conda activate gihoon  

python inversion.py --image_path "/shared/s2/lab01/gihoon/prompt-to-prompt/example_images/gnochi_mirror.jpeg" --prompt "a cat sitting next to a mirror" --out_dir "outout_example_cat"