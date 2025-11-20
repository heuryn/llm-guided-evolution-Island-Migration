#!/bin/bash
#SBATCH --job-name=server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "A100-80GB|H100|H200"
#SBATCH --mem 160G
#SBATCH -c 16

echo "launching LLM Server"

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" >> "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME"

uv run python server.py --host $SERVER_HOSTNAME --port 8137 --workers 1 --model_path "/storage/ice-shared/vip-vvk/llm_storage/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# uv run python server.py --host $SERVER_HOSTNAME --port 8137 --workers 1 --model_path "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/"

# uv run python server.py --host $SERVER_HOSTNAME --port 8137 --workers 1 --model_path "/storage/ice-shared/vip-vvk/llm_storage/Qwen/Qwen2.5-72B-Instruct"

echo "Started LLM Server"

