#!/bin/bash
#SBATCH --job-name=tae_compare
#SBATCH --partition=main
#SBATCH --account=sdibbo_grp
#SBATCH --qos=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --dependency=afterok:51499221:51493916
#SBATCH --output=/scratch/jppatton/langea/logs/tae-compare-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae-compare-%j.err
set -euo pipefail
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/logs /scratch/jppatton/langea/tmp
PY=/bighome/jppatton/langea/runtime/vllm/bin/python
$PY /scratch/jppatton/langea/deploy/tae_2026/compare_geometry_models.py \
  --geometry \
  /scratch/jppatton/langea/runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v2_controls/results.json \
  /scratch/jppatton/langea/runs/tae_2026/llama31_70b_analysis/llama31_70b_20260816/geometry/results.json \
  /scratch/jppatton/langea/runs/tae_2026/llama33_70b_parent_analysis/llama33_70b_20260816/geometry/results.json \
  --equal-n \
  /scratch/jppatton/langea/runs/tae_2026/equal_n/equal_n_20260815_8b_all_layers_v1/results.json \
  /scratch/jppatton/langea/runs/tae_2026/llama31_70b_analysis/llama31_70b_20260816/equal_n/results.json \
  /scratch/jppatton/langea/runs/tae_2026/llama33_70b_parent_analysis/llama33_70b_20260816/equal_n/results.json \
  --output /scratch/jppatton/langea/runs/tae_2026/model_comparison/model_comparison_20260816/results.json \
  --figures /scratch/jppatton/langea/runs/tae_2026/model_comparison/model_comparison_20260816/figures
