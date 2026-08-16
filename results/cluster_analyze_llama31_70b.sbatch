#!/bin/bash
#SBATCH --job-name=tae31_analysis
#SBATCH --partition=main
#SBATCH --account=sdibbo_grp
#SBATCH --qos=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --dependency=afterok:51499182
#SBATCH --output=/scratch/jppatton/langea/logs/tae31-analysis-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae31-analysis-%j.err
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
MANIFEST=/scratch/jppatton/langea/runs/tae_2026/llama31_70b_validation/llama31_70b_validation_20260816/archive_manifest.json
ROOT=/scratch/jppatton/langea/runs/tae_2026/llama31_70b_analysis/llama31_70b_20260816
mkdir -p /scratch/jppatton/langea/runs/tae_2026/llama31_70b_analysis
ARCH=$($PY -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="validated_archive_selected"; print(p["selected_all_layers"])' "$MANIFEST")
VALIDATION=$($PY -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["validation"]["activation_archive"])' "$MANIFEST")
$PY /scratch/jppatton/langea/deploy/tae_2026/validate_activation_archive.py \
  --items "$ITEMS" --activations "$VALIDATION" \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --revision 1605565b47bb9346c5515c34102e054115b4f98b \
  --output "$ROOT/archive_validation.json"
$PY /scratch/jppatton/langea/deploy/tae_2026/run_geometry_all.py \
  --items "$ITEMS" --activations "$ARCH" \
  --output "$ROOT/geometry/results.json" --run-id llama31_70b_geometry_20260816 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --revision 1605565b47bb9346c5515c34102e054115b4f98b \
  --git-commit 2037dd9 --controls
$PY /scratch/jppatton/langea/deploy/tae_2026/run_equal_n_all.py \
  --items "$ITEMS" --activations "$ARCH" \
  --output "$ROOT/equal_n/results.json" --run-id llama31_70b_equal_n_20260816 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --revision 1605565b47bb9346c5515c34102e054115b4f98b \
  --git-commit 2037dd9
$PY -c 'import json,hashlib, pathlib; root=pathlib.Path("'"$ROOT"'"); payload={"status":"analysis_complete","run_id":"llama31_70b_analysis_20260816","job_id":"'"${SLURM_JOB_ID}"'","geometry_sha256":hashlib.sha256((root/"geometry"/"results.json").read_bytes()).hexdigest(),"equal_n_sha256":hashlib.sha256((root/"equal_n"/"results.json").read_bytes()).hexdigest()}; (root/"analysis_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")'
