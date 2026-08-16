#!/bin/bash
#SBATCH --job-name=tae_causal8b_q4
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae-causal8b-q4-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae-causal8b-q4-%j.err
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
OUT=/scratch/jppatton/langea/runs/tae_2026/causal/causal_20260816_8b_quant4_exploratory/baseline.json
set +e
$PY /scratch/jppatton/langea/deploy/tae_2026/run_causal_baseline_quantized.py \
  --items "$ITEMS" \
  --output "$OUT" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --quant-bits 4 --batch-size 4 --max-length 2048 --device-map auto
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  FAIL=/scratch/jppatton/langea/runs/tae_2026/causal/causal_20260816_8b_quant4_exploratory_job${SLURM_JOB_ID}_failed
  mkdir -p "$FAIL"
  $PY -c 'import json,os,sys; json.dump({"schema_version":1,"status":"technical_failure_before_exploratory_quantized_baseline","job_id":os.environ.get("SLURM_JOB_ID"),"exit_code":int(sys.argv[1]),"model":"meta-llama/Llama-3.1-8B-Instruct","revision":"0e9e39f249a16976918f6564b8830bc894c89659","quantization":"bitsandbytes-nf4-4bit","input_sha256":"ebac61b7e6b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022","scientific_boundary":"No exploratory quantized baseline result exists."}, open(sys.argv[2],"w"), indent=2)' "$RC" "$FAIL/failure.json"
  exit "$RC"
fi
