# 8B causal baseline failure

Run ID: `causal_20260816_8b_baseline_job51503230_failed`

Status: technical failure before baseline inference.

The independent one-H100 job requested the frozen model `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`. Hugging Face returned HTTP 403 while loading the tokenizer/config for the gated exact revision. No model weights loaded, no endpoint logits were produced, and no purpose or format accuracy/AUC exists for this run.

The approved credential authenticates to Hugging Face and reports `canReadGatedRepos`, but repository file access for this exact gated model revision is not authorized. The approved local Hugging Face caches contain no exact Llama-3.1-8B snapshot. This is an access failure, not a null causal result.

The transport job was dependency-gated on this baseline and therefore did not run. No frozen protocol parameter was changed.
