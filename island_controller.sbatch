#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=16:00:00
#SBATCH --output=run_job_outputs/Report-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12
module load anaconda3

conda activate llmIslandsEnv
conda info

# LLM_Storage on PACE-ICE
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export LD_LIBRARY_PATH=~/.conda/envs/llmIslandsEnv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Change to the directory where the script is located (replace <username> with your actual username)
cd /home/hice1/<username>/scratch/llm-guided-evolution-Island-Migration

# Starts running Island Migration (This is the controller script)
python islands_wrapper.py test7 --num_islands 2
