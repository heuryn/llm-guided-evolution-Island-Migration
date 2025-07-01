#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=16:00:00
#SBATCH --output=run_job_outputs/Report-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12

# LLM_Storage on PACE-ICE
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Change to the directory where the script is located (replace <username> with your actual username)
cd /home/hice1/<username>/scratch/llm-guided-evolution-Island-Migration

# Starts running Island Migration (This is the controller script)
uv run python islands_wrapper.py <directory_name> --num_islands 1
