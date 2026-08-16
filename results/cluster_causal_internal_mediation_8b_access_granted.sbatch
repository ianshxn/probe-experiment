#!/bin/bash
#SBATCH --job-name=tae_causal8b_internal_access
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae-causal8b-internal-access-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae-causal8b-internal-access-%j.err
set -euo pipefail
source /home/jppatton/langea/credentials/huggingface.env
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/logs /scratch/jppatton/langea/tmp /scratch/jppatton/langea/runs/tae_2026/causal
PY=/bighome/jppatton/langea/runtime/vllm/bin/python
ITEMS=/scratch/jppatton/langea/deploy/tae_2026/rendered_items_canonical_lf.jsonl
PROTOCOL=/scratch/jppatton/langea/deploy/tae_2026/causal_20260816_internal_mediation_protocol.json
OUT=/scratch/jppatton/langea/runs/tae_2026/causal/causal_20260816_internal_mediation_access_granted
$PY /scratch/jppatton/langea/deploy/tae_2026/run_causal_internal_mediation.py \
  --items "$ITEMS" \
  --output "$OUT" \
  --protocol "$PROTOCOL" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --batch-size 8 --max-length 2048 --device-map auto
