#!/bin/bash
#SBATCH --job-name=Infer
#SBATCH --nodes=1

echo "launching LLM Server"

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1

export SERVER_HOSTNAME=$(hostname)

echo "Testing Inference Server on: $SERVER_HOSTNAME"

uv run python test_server_inference.py

echo "End"
