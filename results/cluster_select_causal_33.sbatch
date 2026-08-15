#!/bin/bash
#SBATCH --job-name=tae33_select
#SBATCH --partition=main
#SBATCH --account=sdibbo_grp
#SBATCH --qos=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --dependency=afterok:51493916
#SBATCH --output=/scratch/jppatton/langea/logs/tae33-select-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae33-select-%j.err
set -euo pipefail
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/logs /scratch/jppatton/langea/tmp
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/deploy/tae_2026/select_causal_layers.py \
  --geometry /scratch/jppatton/langea/runs/tae_2026/llama33_70b_parent_analysis/llama33_70b_20260816/geometry/results.json \
  --output /scratch/jppatton/langea/runs/tae_2026/causal/causal_20260816_33_geometry_frozen/protocol.json \
  --run-id causal_20260816_33_geometry_frozen --git-commit e5bf38b
