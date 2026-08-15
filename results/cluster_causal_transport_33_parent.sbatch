#!/bin/bash
#SBATCH --job-name=tae33_transport
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --dependency=afterok:51495134
#SBATCH --output=/scratch/jppatton/langea/logs/tae33-transport-parent-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae33-transport-parent-%j.err
set -euo pipefail
source /home/jppatton/langea/credentials/huggingface.env
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/logs /scratch/jppatton/langea/tmp
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/deploy/tae_2026/run_causal_transport.py \
  --items /scratch/jppatton/langea/deploy/tae_2026/rendered_items_canonical_lf.jsonl \
  --output /scratch/jppatton/langea/runs/tae_2026/causal/causal_20260816_33_geometry_frozen/transport_33 \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --protocol /scratch/jppatton/langea/runs/tae_2026/causal/causal_20260816_33_geometry_frozen/protocol.json \
  --batch-size 2 --max-length 2048 --device-map auto --local-files-only
