# Failed causal execution

Run `causal_20260815_8b_job51488273_failed` failed before baseline inference. The approved cluster credential was present as `HF_TOKEN`, but the Hugging Face request for the gated exact revision returned HTTP 403 (`GatedRepoError`). No model weights loaded, so no baseline, steering, or causal result exists.

The frozen protocol remains unchanged at `runs/tae_2026/causal/causal_20260815_geometry_frozen/`. The stderr transcript is preserved in `job.err`. This is a technical access failure, not a scientific null or negative result. Any future retry requires an authorized exact Llama-3.1-8B checkpoint or an explicitly documented alternative model; it must not be relabeled as this run.
