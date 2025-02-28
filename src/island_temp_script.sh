#!/bin/bash
#SBATCH --job-name=LLM_Island_gemma2
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=08:00:00
#SBATCH -oReport_islands-%j.out
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
python islandIntegration.py first_test/island_gemma2 --llm_model gemma2
