#!/bin/bash
#SBATCH --job-name=tae33_analysis
#SBATCH --partition=main
#SBATCH --account=sdibbo_grp
#SBATCH --qos=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --dependency=afterok:51493815
#SBATCH --output=/scratch/jppatton/langea/logs/tae33-analysis-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae33-analysis-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export PYTHONPATH=/scratch/jppatton/langea/deploy/tae_2026
mkdir -p /scratch/jppatton/langea/logs /scratch/jppatton/langea/tmp
PY=/bighome/jppatton/langea/runtime/vllm/bin/python
ITEMS=/scratch/jppatton/langea/deploy/tae_2026/rendered_items_canonical_lf.jsonl
ARCH=/scratch/jppatton/langea/runs/tae_2026/llama33_70b_parent/llama33_70b_20260816_rev6f6073b/activations_all_layers.npz
ROOT=/scratch/jppatton/langea/runs/tae_2026/llama33_70b_parent_analysis/llama33_70b_20260816
$PY /scratch/jppatton/langea/deploy/tae_2026/validate_activation_archive.py \
  --items "$ITEMS" \
  --activations "${ARCH%_all_layers.npz}.npz" \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --output "$ROOT/archive_validation.json"
$PY /scratch/jppatton/langea/deploy/tae_2026/run_geometry_all.py \
  --items "$ITEMS" --activations "$ARCH" \
  --output "$ROOT/geometry/results.json" --run-id llama33_70b_geometry_20260816 \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --git-commit 09115a7 --controls
$PY /scratch/jppatton/langea/deploy/tae_2026/run_equal_n_all.py \
  --items "$ITEMS" --activations "$ARCH" \
  --output "$ROOT/equal_n/results.json" --run-id llama33_70b_equal_n_20260816 \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --git-commit 09115a7
$PY -c 'import json,hashlib,pathlib; root=pathlib.Path("'"$ROOT"'"); payload={"status":"analysis_complete","run_id":"llama33_70b_parent_analysis_20260816","job_id":"'"${SLURM_JOB_ID}"'","geometry_sha256":hashlib.sha256((root/"geometry"/"results.json").read_bytes()).hexdigest(),"equal_n_sha256":hashlib.sha256((root/"equal_n"/"results.json").read_bytes()).hexdigest()}; (root/"analysis_manifest.json").parent.mkdir(parents=True,exist_ok=True); (root/"analysis_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")'
