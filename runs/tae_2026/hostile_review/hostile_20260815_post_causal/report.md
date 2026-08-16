# Hostile reproducibility review: post-causal status

Run ID: `hostile_20260815_post_causal`
Git commit: `790b144f9786fe49f35513c332e3d09dfd503dc3`
Status: **critical findings remain open**.

## Closed or bounded findings

- The original pooled-versus-transfer row-count confound is fixed for the 8B primary artifact by the validated all-layer equal-N analysis. Benchmark-only, casual-only, and equal-N mixed-format training use equal training-row counts.
- Full-depth 8B raw geometry and fixed-dimensional controls are present, including fixed 256-D random projections and PCA controls.
- The factorial residual intervention protocol is frozen before outcomes, uses held-out purpose-family fitting, raw-space directions, zero-strength/sign controls, and 32 fixed random directions.
- The direct endpoint gate was actually executed after access approval. Purpose accuracy/AUC was `0.4757/0.4747`, below the frozen accuracy threshold `0.55`; behavioral transport was therefore refused. The separate internal-mediation fallback completed and is reported only as internal mediation.
- Current archive semantics are guarded: zero-based transformer-block output before final model norm, last prompt token after chat templating, BF16 inference, float32 storage.

## Critical open findings

1. The canonical bank contains 288 rows / 72 complete payload blocks, while design notes describe an intended 320 / 80 grid. The missing-block provenance and coverage implications remain unresolved.
2. Target-format exposure is characterized by a separate trajectory, but it is not interchangeable with the equal-N mixed-format estimand. Transfer claims must state which exposure condition is being used.
3. Family uncertainty remains conditional on fixed fold fits. No refit bootstrap is promoted to confirmation-grade evidence.
4. Exact Llama-3.1-70B extraction job `51479700` remains pending for resources; no exact archive or three-model result exists.
5. Parent Llama-3.3-70B extraction `51493815` remains pending priority; dependent analysis, model comparison, and 3.3 causal jobs remain gated.
6. The direct behavioral causal endpoint is unusable on this wrapper/mapping. The internal result cannot be upgraded to behavioral steering or generated-behavior evidence.

## Required delivery gates

- Validate a user-owned or canonical-retry 3.1-70B archive against exact input bytes, item order, model revision, layer IDs, and dtype semantics.
- Complete all-layer geometry/equal-N analysis for 3.1-70B and parent 3.3-70B, then run the frozen matched comparison without changing estimands after inspection.
- Retain the null behavioral endpoint result and internal-mediation boundary in the final paper.
- Do not make a three-model, multilingual, naturalistic, or behavioral causal claim before the corresponding artifact exists and passes provenance checks.
