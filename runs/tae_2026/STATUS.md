# TAE 2026 Sprint Status

HEAD: `be8c5d6` (BF16 causal replication, natural-target controls, and paper update)
INPUT SHA: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022` (288 items; canonical `runs/tae_2026/input/rendered_items_canonical_lf.jsonl`)
MODELS: Llama-3.1-8B-Instruct (revision `0e9e39f249a16976918f6564b8830bc894c89659`, access preflight passed); Llama-3.3-70B exact parent extraction job `51493815` (legacy result retained only as development evidence); Llama-3.1-70B revision `1605565b47bb9346c5515c34102e054115b4f98b` (access preflight still blocked, pending user-owned job).
- ACTIVE JOBS: user-owned `51479700` and `51484306` untouched; parent-owned 3.1 validation/recovery `51499967`, 3.1 analysis `51500106`, matched comparison `51500207`, 3.3 extraction/analysis `51493815`/`51493916`; parent-owned causal 3.3 selector/baseline/transport jobs `51494932`, `51495134`, `51495235`; completed 8B baseline `51515708`, refused transport `51515710`, completed internal mediation `51521619`, independent granted-access baseline `51535654`, and independent internal mediation `51536964`.

## EQUAL-N

- status: **all-layer validated equal-N and target-format exposure trajectory complete for 8B**.
- main numbers: layer-24 benchmark→benchmark/casual `0.7444/0.5544` AUC, casual→casual/benchmark `0.7095/0.5656`, equal-N mixed `0.6684`; across layers interaction ratio vs equal-N mixed AUC `-0.1605`, vs benchmark→casual `-0.3818`, vs casual→benchmark `-0.3474`; mixed AUC range `0.5505–0.9237`.
- artifact: `runs/tae_2026/equal_n/equal_n_20260815_8b_all_layers_v1/results.json` and `report.md`; result SHA256 `f064b230f7076562bf68033d03129d890020ef357cdb0b70b0dd143983d711ed`.
## GEOMETRY
- status: **all-layer raw/fixed-dimensional geometry, conditional family bootstrap, and equal-N target-exposure curves complete for 8B**.
- main numbers: raw ratio range `0.2780–0.8428`; PCA ratio range `0.2349–0.8224`; raw/PCA ratio correlation `0.9993`; PCA ratio vs benchmark→casual AUC `-0.3662`; random-projection ratio vs AUC `-0.3795`; raw ratio vs AUC `-0.3814`; B-projection vs AUC−BA gap `0.2286`.
- artifact: bootstrap `runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v3_bootstrap/`; source controlled geometry `runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v2_controls/`.
- offline mechanism artifact: `runs/tae_2026/geometry/counterfactual_reconstruction_20260816_8b_final/`; L5 mean purpose C benefit `0.000402`, L16 `0.005555`, paired L16-minus-L5 `0.005153` with 95% CI `[-0.000347, 0.010752]`; PCA controls L5/L16 `0.000551/0.007288`, random controls `0.000587/0.004730`; status is `DEVELOPMENT_OFFLINE_MECHANISM`, not causal steering.

## CAUSAL
- status: **final BF16 internal causal replication and natural-target control run complete; behavioral transport remains correctly refused after the frozen purpose endpoint gate failed.**
- exact BF16 model: Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`; input SHA `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`.
- independent BF16 baseline job `51535654`: purpose accuracy/AUC `0.4757/0.4747`; format accuracy/AUC `0.5694/0.5988`; purpose gate `<0.55`, so behavioral transport was not run.
- final BF16 internal run job `51575504`: `runs/tae_2026/causal/causal_20260816_8b_final_bf16_internal/`; results SHA256 `1f1c54293ac8d2ecf099eb81e713809376b867319d793e26db21a22540f4b58d`; protocol and report are co-located.
- final projection readout at L24: full-minus-A-only purpose projection `+0.001585` at L5 and `-0.012939` at L16; paired L16-minus-L5 `-0.014524`, five-family bootstrap 95% CI `[-0.017485,-0.012023]`.
- natural-target control job `51580007`: `runs/tae_2026/causal/causal_20260816_8b_bf16_natural_counterfactual_controls/`; 65,664 records, 32 fixed random orthogonal-C controls, results SHA256 `5686e9b6563044c7adb1e9492cc4b7f4ff200192986eaf81ca96b0ec527f17e1`.
- natural target at L16/L24: full target distance `1.833561`, A-only `1.840693`, zero `1.850649`; full-minus-A-only improvement `+0.007132` at L16 versus `-0.000518` at L5; complete-payload bootstrap CI `[+0.002978,+0.012349]`.
- natural-target controls at L16: wrong-sign C incremental `-0.011023`, norm-matched A-only `+0.002779`, shuffled C `-0.002543`; random-bank L16-minus-L5 incremental mean `-0.001825` (SD `0.001402`), observed `+0.007650` exceeds all 32 controls.
- scientific boundary: this is internal causal mediation of factorial transport, not generated-behavior steering; exact 3.3-70B causal path remains dependency-gated.
## 3.1-70B
- status: **submitted; preflight with the current approved HF credential returns HTTP 403 / “request ... awaiting a review from the repo authors” for the exact gated revision. No local exact 3.1-70B cache exists; user-owned extraction job `51479700` has not executed, so its execution outcome remains unobserved.**
- jobs: user-owned extraction `51479700`; parent-owned validator/recovery `51499967` is now explicitly `afterany:51479700`, so a failed gated-model extraction still triggers the parent-owned canonical retry; parent-owned analysis `51500106`; all remain dependency-gated. The validator rejects noncanonical input bytes and, if needed, extracts a new immutable archive from the canonical bank without modifying the user-owned run.
- artifacts: user-owned archive target `runs/tae_2026/llama31_70b/llama31_70b_20260815_rev1605565b/`; canonical retry target `runs/tae_2026/llama31_70b_canonical/llama31_70b_20260816_rev1605565b/`; validation manifest target `runs/tae_2026/llama31_70b_validation/llama31_70b_validation_20260816/archive_manifest.json`.

## MODEL COMPARISON
- status: partial ledger complete; 8B/3.3-70B exact revisions repaired; 3.1-70B pending user-owned extraction.
- dependent analysis: parent-owned `51500106` consumes only the validated archive manifest; matched comparison `51500207` waits for `51500106` and exact 3.3 analysis `51493916`.
- queue evidence (current): `51479700` and `51484306` are pending with estimated start `2026-08-17T23:30:18`; parent `51493815` is pending with estimated start `2026-08-18T12:52:23`; `51499967` is pending on `afterany:51479700`; no output directories contain archives/results yet.

## HOSTILE REVIEW
- open critical findings: target-format exposure estimand; 72-versus-80 coverage provenance; conditional (not refit) family bootstrap; behavioral endpoint blocked after near-chance baseline; pending 3.1-70B and 3.3 parent extraction; legacy layer semantics guarded. Fixed: 8B full-depth fixed-dimensional controls and internal-mediation protocol/result boundary.

## PAPER
- pages drafted: anonymized workshop paper with integrated equal-N/geometry results, internal mediation result, paper-local geometry/partial-model figures, related-work matrix, and bibliography under `paper/tae_2026_workshop/`.
- missing evidence: matched three-model results; behavioral causal transport remains intentionally unavailable because the direct endpoint gate failed.
- figures: geometry full-depth curves and partial 8B/3.3-70B ledger figures embedded; causal internal mediation and matched three-model figures remain unavailable as standalone figures.
- compile check: official January 2026 `neurips_2026.sty` is vendored from the public NeurIPS 2026 style source; `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` completed successfully (6 pages, no unresolved citations on final pass).

## CURRENT BEST DEFENSIBLE CLAIM
- On the 288-item Llama-3.1-8B layer-24 development artifact, equal-N mixed-format training AUC `0.6684` exceeds benchmark-only `0.6113` and casual-only `0.6259`, while pooled training is `0.7029`; the gap is not sample-size-only on this artifact. Across raw all-layer geometry, `norm(C)/norm(A)` versus benchmark→casual AUC correlates `-0.3814`, but this is one-model trajectory evidence, not independent-N significance or causal evidence.
- BF16 internal causal replication on the exact 8B revision: natural-target distance at L16/L24 was `1.833561` for full `A+fC`, `1.840693` for A-only, and `1.850649` for zero; full-minus-A-only improvement was `+0.007132` at L16 versus `-0.000518` at L5, with complete-payload bootstrap 95% CI `[+0.002978,+0.012349]`. This is internal causal mediation only because the direct behavioral endpoint failed validation.

## INPUT SERIALIZATION
- The canonical UTF-8/LF bank is frozen at `runs/tae_2026/input/rendered_items_canonical_lf.jsonl`, SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`. The deployed legacy `rendered_items.jsonl` has identical parsed rows/order but CRLF serialization (`4780e19e...`); new parent-owned jobs use the canonical bank. User-owned jobs were not modified; their resulting archive must pass exact semantic and byte-level provenance checks before inclusion.

## NEXT BLOCKER
- Validate the user-owned 3.1-70B archive or complete the canonical retry, then await parent-owned 3.3 analyses; the 8B internal mediation result is complete and bounded to internal readouts, while the exact cached 3.3 causal path remains active.
