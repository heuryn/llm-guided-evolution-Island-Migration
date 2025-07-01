# ROOT DIR in ICEHAMMER, change <username> to your actual username
ROOT_DIR = "/home/<username>/llm-guided-evolution-Island-Migration/"

# All GPUs available on ICEHAMMER (in order of performance)
LLM_GPU = 'NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100S-PCIE-32GB|TeslaV100-PCIE-32GB|TITANRTX|GeForceGTXTITANX|GeForceGTX1080Ti|QuadroRTX4000|QuadroP4000|TeslaK40c|TeslaK40m|TeslaK20Xm|TeslaK20c|TeslaK20m'

#: Template script for submitting job for evaluation
PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 0-06:00
#SBATCH -C "{}"
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --mem 80G
#SBATCH --output=run_job_outputs/evaluation/slurm-%j.out

echo "Launching AIsurBL"
hostname
# Load GCC version 9.2.0
# module load gcc/13.2.0
module load cuda/12
module load anaconda3

# Activate Conda environment
conda activate {}

# conda info
# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export MKL_THREADING_LAYER=GNU

# Run Python script
{}
"""

#: Template script for submitting a prompt to the LLM
LLM_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={}
#SBATCH -t 0-03:00
#SBATCH -C "{}"
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --mem 32G
#SBATCH --output=run_job_outputs/evolution/slurm-%j.out

echo "Launching AIsurBL"
hostname

module load cuda/12
module load anaconda3
# Activate Conda environment
conda activate {}
# conda info

CUDA_LAUNCH_BLOCKING=1

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
{}
"""

#: Template script for submitting an island run
ISLANDS_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=LLM_Island_{}
#SBATCH -t 5-00:00
#SBATCH -C "{}"
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --mem 80G
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12
module load anaconda3

conda activate {}
conda info

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
python run_improved.py {} --global_path {} --llm_model {}
"""