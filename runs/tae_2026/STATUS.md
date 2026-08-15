# TAE 2026 Sprint Status

HEAD: `9ceff263a768ed0dd0a840cf3162e456ed4499f7`
INPUT SHA: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022` (288 items; `probe results/llama31_8b_3f9ccbe890cb7a06/meta.jsonl`)
MODELS: Llama-3.1-8B-Instruct (revision `0e9e39f249a16976918f6564b8830bc894c89659`); Llama-3.3-70B local result (revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b`, recorded from extraction config; matched analysis provenance still being repaired); Llama-3.1-70B revision `1605565b47bb9346c5515c34102e054115b4f98b` pending extraction.
- ACTIVE JOBS: SLURM `51479700` (`tae31_70b`) pending priority on `gpu`, `4xh100-80`; exact 3.1-70B BF16 all-layer extraction.

## EQUAL-N

- status: **all-layer validated equal-N and target-format exposure trajectory complete for 8B**.
- main numbers: layer-24 benchmark→benchmark/casual `0.7444/0.5544` AUC, casual→casual/benchmark `0.7095/0.5656`, equal-N mixed `0.6684`; across layers interaction ratio vs equal-N mixed AUC `-0.1605`, vs benchmark→casual `-0.3818`, vs casual→benchmark `-0.3474`; mixed AUC range `0.5505–0.9237`.
- artifact: `runs/tae_2026/equal_n/equal_n_20260815_8b_all_layers_v1/results.json` and `report.md`; result SHA256 `f064b230f7076562bf68033d03129d890020ef357cdb0b70b0dd143983d711ed`.
## GEOMETRY
- status: **all-layer raw/fixed-dimensional geometry, conditional family bootstrap, and equal-N target-exposure curves complete for 8B**.
- main numbers: raw ratio range `0.2780–0.8428`; PCA ratio range `0.2349–0.8224`; raw/PCA ratio correlation `0.9993`; PCA ratio vs benchmark→casual AUC `-0.3662`; random-projection ratio vs AUC `-0.3795`; raw ratio vs AUC `-0.3814`; B-projection vs AUC−BA gap `0.2286`.
- artifact: bootstrap `runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v3_bootstrap/`; source controlled geometry `runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v2_controls/`.

## CAUSAL
- status: **geometry-only protocol frozen; baseline endpoint blocked before steering**.
- baseline endpoint: not run; no local model endpoint, exact 3.1-70B job pending. No steering outcomes inspected.
- selected layers: shared `5`, interaction `16`, selected only from all-layer geometry (eligibility pooled AUC >= 0.80).
- protocol hash: `d651da40a535526d382936a42428aa5ce0e816d1ff20cba8b79f55dadec140c9`.
- main effects: not measured.
- artifact: `runs/tae_2026/causal/causal_20260815_geometry_frozen/` (`protocol.json`, `baseline.json`, `report.md`).

## 3.1-70B
- status: **submitted; pending cluster execution**. Exact Hub revision resolved as `1605565b47bb9346c5515c34102e054115b4f98b`; no matching cache existed before submission.
- job: SLURM `51479700`, `tae31_70b`, `gpu`/`gpu` QoS, `4xh100-80`, pending priority. Job script and runner are `results/cluster_extract_tae.py` and `results/cluster_extract_tae.sbatch`; remote copies live only under approved `/scratch/jppatton/langea/...` paths.
- artifact: pending immutable `runs/tae_2026/llama31_70b/llama31_70b_20260815_rev1605565b/`; exact input hash and BF16/all-layer/pre-final-norm config are frozen.

## MODEL COMPARISON
- status: current local 8B/3.3-70B development evidence only.
- key result: existing comparison is confounded by generation/training and lacks matched 3.1-70B.

## HOSTILE REVIEW
- open critical findings: target-format exposure control; 72-versus-80 coverage provenance; conditional (not refit) family bootstrap; full-depth fixed-dimensional controls; causal endpoint baseline; pending 3.1-70B extraction; legacy layer semantics guarded.

## PAPER
- pages drafted: anonymized skeleton, section placeholders, related-work matrix, and bibliography under `paper/tae_2026_workshop/`; page count not yet compiled.
- missing sections: finalized observational/causal/model results and limitations evidence.
- missing figures: causal transport and three-model comparison; geometry full-depth figures now present under `runs/tae_2026/figures/geometry_20260815_8b_all_layers/`.

## CURRENT BEST DEFENSIBLE CLAIM
- On the 288-item Llama-3.1-8B layer-24 development artifact, equal-N mixed-format training AUC `0.6684` exceeds benchmark-only `0.6113` and casual-only `0.6259`, while pooled training is `0.7029`; the gap is not sample-size-only on this artifact. Across raw all-layer geometry, `norm(C)/norm(A)` versus benchmark→casual AUC correlates `-0.3814`, but this is one-model trajectory evidence, not independent-N significance or causal evidence.

## NEXT BLOCKER
- Monitor SLURM `51479700`; then run the already-frozen direct-logit endpoint baseline before any steering and complete matched 3.1-70B analyses.
