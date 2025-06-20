#!/bin/bash
#SBATCH --job-name=LLM_Island_mixtral
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=16:00:00
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
python run_improved.py test7/island_mixtral --global_path test7/global_data --llm_model mixtral
