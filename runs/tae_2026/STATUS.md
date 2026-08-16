# TAE 2026 Sprint Status

HEAD: `adc9f43` (independent granted-access 8B baseline, gate-refused behavioral transport, and reproduced internal mediation)
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

## CAUSAL
- status: **new HF access preflight is `ACCESS_OK` for the exact 8B revision. Independent baseline job `51535654` completed on one H100; purpose endpoint remains near chance, so behavioral transport was correctly refused by the frozen gate. Independent internal-mediation fallback job `51536964` completed without dependencies and reproduced the prior internal result.**
- 3.3 protocol selection: geometry-only selector `51494932` waits on 3.3 analysis; baseline `51495134` and transport `51495235` are dependency-gated. No 3.3 endpoint or steering outcome has been inspected.
- 8B prior baseline job `51503230`: one H100, failed before tokenizer/model loading with HTTP 403. Prior transport `51503331` was canceled; neither produced a scientific outcome.
- 8B retry baseline job `51515708`: completed; immutable artifact `runs/tae_2026/causal/causal_20260816_8b_baseline_retry/baseline.json`, SHA256 `d92c3f481aeafb8c0cd9defa77da4441b38ca29bfe4942309dba677f7d32b5fe`.
- 8B endpoint metrics: purpose accuracy/AUC `0.4757/0.4747`; format accuracy/AUC `0.5694/0.5988`; candidate mapping seed `2026081533`. Purpose endpoint is below the frozen `0.55` accuracy threshold.
- Independent granted-access baseline job `51535654` (one `H100-80GB`, no dependency) reproduced the same endpoint metrics; artifact `runs/tae_2026/causal/causal_20260816_8b_baseline_access_granted/baseline.json`, SHA256 `6d68379f35dcff9413ca9b228c800def0af19993a363627a2a4b11eb8476635d`.
- Granted-access behavioral transport was not run after the predeclared gate failed; failure artifact `runs/tae_2026/causal/causal_20260816_8b_transport_access_granted_gate_failed/failure.json`, SHA256 `a1d2463f163b4eaac5a2a0683231932d233a7491d760013c1a9ed16d750f1294`.
- Internal mediation setup job `51520918` failed before loading the protocol path and produced no scientific output. Corrected retry job `51521619` completed with fixed intervention layers `5/16`, downstream readout layers `16/24/31`, all frozen modes and random controls; output `runs/tae_2026/causal/causal_20260816_internal_mediation_retry/`.
- 8B selected layers: shared `5`, interaction `16`, selected only from all-layer geometry (eligibility pooled AUC >= 0.80).
- 8B protocol hash: `d651da40a535526d382936a42428aa5ce0e816d1ff20cba8b79f55dadec140c9`.
- frozen protocol artifact remains `runs/tae_2026/causal/causal_20260815_geometry_frozen/`; internal fallback protocol SHA256 `3f5b2db1e8774b45c2111d63c25dfa2c52ee6fc2f7b2478759a7f9a8b8bb5372`; completed results SHA256 `887a6d7b0d01c60d9f4fe4009ccbe2d80565b51d5d6549209d8cfd6da523a4b8`.
- Access preflight artifact: `runs/tae_2026/causal/causal_20260816_8b_access_granted_preflight/access.json`, SHA256 `e6c911bb7366c8d78f4cf545af76d3f84a25e3405f794735809d4c9e23bc137c`; independent internal batch SHA256 `cacc377d251c35fdb5123a885c09c9c82f3fa5dc9c6ce7927b259b4cbd3e1357`.
- Independent internal bundle: `runs/tae_2026/causal/causal_20260816_internal_mediation_access_granted/`; protocol SHA256 `5937f98ff37aa765370186fdb1f09041a824d41070ae2b3eca8b64e833948299`; results SHA256 `887a6d7b0d01c60d9f4fe4009ccbe2d80565b51d5d6549209d8cfd6da523a4b8`; report SHA256 `037fd409386a4c413342ff6b3394bf49dfc22e5fea8be1bfdd8a8c802c7ef2d7`. The full factorial purpose transport at L16 was `-0.0796` versus `-0.0667` for A-only; off-target format movement was `+0.0005`.

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
- At layer 16, frozen held-out-family residual transport changed the downstream purpose projection by `-0.0796` with full `A+C` transport versus `-0.0667` with `A` only, and the downstream format projection by `-0.2004` with full `B+C` transport versus `-0.1985` with `B` only; this is internal mediation only because the direct behavioral endpoint failed validation.

## INPUT SERIALIZATION
- The canonical UTF-8/LF bank is frozen at `runs/tae_2026/input/rendered_items_canonical_lf.jsonl`, SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`. The deployed legacy `rendered_items.jsonl` has identical parsed rows/order but CRLF serialization (`4780e19e...`); new parent-owned jobs use the canonical bank. User-owned jobs were not modified; their resulting archive must pass exact semantic and byte-level provenance checks before inclusion.

## NEXT BLOCKER
- Validate the user-owned 3.1-70B archive or complete the canonical retry, then await parent-owned 3.3 analyses; the 8B internal mediation result is complete and bounded to internal readouts, while the exact cached 3.3 causal path remains active.
