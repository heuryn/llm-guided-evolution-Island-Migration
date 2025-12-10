#!/bin/bash
#SBATCH --job-name=Server_Initializer
#SBATCH --output=run_job_outputs/Server_Initializer-%j.out


module load cuda
module load uv

HOSTNAME_FILE=$(pwd)"/hostname.log"

echo "Clearing $HOSTNAME_FILE"
> "$HOSTNAME_FILE"
echo "Initializing LLM Servers"

uv run python islands_server_initializer.py

echo "Initialized All LLM Server Jobs"