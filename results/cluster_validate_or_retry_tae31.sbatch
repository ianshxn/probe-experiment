#!/bin/bash
#SBATCH --job-name=tae31_validate
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=3-00:00:00
#SBATCH --dependency=afterok:51479700
#SBATCH --output=/scratch/jppatton/langea/logs/tae31-validate-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae31-validate-%j.err
set -euo pipefail
source /home/jppatton/langea/credentials/huggingface.env
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/cache /scratch/jppatton/langea/tmp /scratch/jppatton/langea/logs
PY=/bighome/jppatton/langea/runtime/vllm/bin/python
ITEMS=/scratch/jppatton/langea/deploy/tae_2026/rendered_items_canonical_lf.jsonl
LEGACY=/scratch/jppatton/langea/runs/tae_2026/llama31_70b/llama31_70b_20260815_rev1605565b/activations.npz
RETRY=/scratch/jppatton/langea/runs/tae_2026/llama31_70b_canonical/llama31_70b_20260816_rev1605565b
MANIFEST=/scratch/jppatton/langea/runs/tae_2026/llama31_70b_validation/llama31_70b_validation_20260816/archive_manifest.json
$PY /scratch/jppatton/langea/deploy/tae_2026/validate_or_retry_tae31.py \
  --items "$ITEMS" \
  --legacy-archive "$LEGACY" \
  --retry-items "$ITEMS" \
  --retry-output "$RETRY" \
  --manifest "$MANIFEST" \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --revision 1605565b47bb9346c5515c34102e054115b4f98b \
  --batch-size 1 --max-length 2048
