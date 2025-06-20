#!/bin/bash
#SBATCH --job-name=LLM_Island_qwen25
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=03:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12
module load anaconda3

conda activate llmIslandsEnv
conda info

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
python islandIntegration.py first_test/island_qwen25 --global_path first_test/global_data --llm_model qwen25
