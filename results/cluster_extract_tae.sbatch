#!/bin/bash
#SBATCH --job-name=tae31_70b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae31_70b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae31_70b-%j.err
set -euo pipefail
source /home/jppatton/langea/credentials/huggingface.env
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/cache /scratch/jppatton/langea/tmp /scratch/jppatton/langea/logs
cd /scratch/jppatton/langea/deploy/tae_2026
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/deploy/tae_2026/cluster_extract_tae.py
