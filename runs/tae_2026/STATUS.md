# TAE 2026 Sprint Status

HEAD: `d415a7f` (exact archive provenance gates and canonical retry resources)
INPUT SHA: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022` (288 items; canonical `runs/tae_2026/input/rendered_items_canonical_lf.jsonl`)
MODELS: Llama-3.1-8B-Instruct (revision `0e9e39f249a16976918f6564b8830bc894c89659`); Llama-3.3-70B exact parent extraction job `51493815` (legacy result retained only as development evidence); Llama-3.1-70B revision `1605565b47bb9346c5515c34102e054115b4f98b` pending user-owned job.
- ACTIVE JOBS: user-owned `51479700` and `51484306` untouched; parent-owned 3.1 validation/recovery `51499967`, 3.1 analysis `51500106`, matched comparison `51500207`, 3.3 extraction/analysis `51493815`/`51493916`; parent-owned causal 3.3 selector/baseline/transport jobs `51494932`, `51495134`, `51495235`; causal 8B `51488273` failed before baseline at gated model access.

## EQUAL-N

- status: **all-layer validated equal-N and target-format exposure trajectory complete for 8B**.
- main numbers: layer-24 benchmark→benchmark/casual `0.7444/0.5544` AUC, casual→casual/benchmark `0.7095/0.5656`, equal-N mixed `0.6684`; across layers interaction ratio vs equal-N mixed AUC `-0.1605`, vs benchmark→casual `-0.3818`, vs casual→benchmark `-0.3474`; mixed AUC range `0.5505–0.9237`.
- artifact: `runs/tae_2026/equal_n/equal_n_20260815_8b_all_layers_v1/results.json` and `report.md`; result SHA256 `f064b230f7076562bf68033d03129d890020ef357cdb0b70b0dd143983d711ed`.
## GEOMETRY
- status: **all-layer raw/fixed-dimensional geometry, conditional family bootstrap, and equal-N target-exposure curves complete for 8B**.
- main numbers: raw ratio range `0.2780–0.8428`; PCA ratio range `0.2349–0.8224`; raw/PCA ratio correlation `0.9993`; PCA ratio vs benchmark→casual AUC `-0.3662`; random-projection ratio vs AUC `-0.3795`; raw ratio vs AUC `-0.3814`; B-projection vs AUC−BA gap `0.2286`.
- artifact: bootstrap `runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v3_bootstrap/`; source controlled geometry `runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v2_controls/`.

## CAUSAL
- status: **3.3-70B causal recovery path queued from exact cached weights; 8B causal path failed before baseline because HF gated access returned HTTP 403**.
- 3.3 protocol selection: geometry-only selector `51494932` waits on 3.3 analysis; baseline `51495134` and transport `51495235` are dependency-gated. No 3.3 endpoint or steering outcome has been inspected.
- 8B baseline endpoint: not run; no 8B steering outcomes inspected or produced.
- 8B selected layers: shared `5`, interaction `16`, selected only from all-layer geometry (eligibility pooled AUC >= 0.80).
- 8B protocol hash: `d651da40a535526d382936a42428aa5ce0e816d1ff20cba8b79f55dadec140c9`.
- 8B failure artifact: `runs/tae_2026/causal/causal_20260815_8b_job51488273_failed/` (failure SHA256 `717bf442bbf2eec7ce714302ac98acd8e06796a8edb3f7772fefe51364fe855d`).
- frozen 8B protocol artifact remains `runs/tae_2026/causal/causal_20260815_geometry_frozen/`; 3.3 results will be a separate immutable run.

## 3.1-70B
- status: **submitted; provenance-gated validation/recovery queued after the user-owned extraction**. Exact Hub revision resolved as `1605565b47bb9346c5515c34102e054115b4f98b`; the user-owned job still has not executed.
- jobs: user-owned extraction `51479700`; parent-owned validator/recovery `51499967`; parent-owned analysis `51500106`; all remain dependency-gated. The validator rejects noncanonical input bytes and, if needed, extracts a new immutable archive from the canonical bank without modifying the user-owned run.
- artifacts: user-owned archive target `runs/tae_2026/llama31_70b/llama31_70b_20260815_rev1605565b/`; canonical retry target `runs/tae_2026/llama31_70b_canonical/llama31_70b_20260816_rev1605565b/`; validation manifest target `runs/tae_2026/llama31_70b_validation/llama31_70b_validation_20260816/archive_manifest.json`.

## MODEL COMPARISON
- status: partial ledger complete; 8B/3.3-70B exact revisions repaired; 3.1-70B pending user-owned extraction.
- dependent analysis: parent-owned `51500106` consumes only the validated archive manifest; matched comparison `51500207` waits for `51500106` and exact 3.3 analysis `51493916`.

## HOSTILE REVIEW
- open critical findings: target-format exposure control; 72-versus-80 coverage provenance; conditional (not refit) family bootstrap; full-depth fixed-dimensional controls; causal endpoint baseline blocked by gated 8B access; pending 3.1-70B extraction; legacy layer semantics guarded.

## PAPER
- pages drafted: anonymized workshop paper with integrated equal-N/geometry results, paper-local geometry/partial-model figures, related-work matrix, and bibliography under `paper/tae_2026_workshop/`.
- missing evidence: causal transport outcomes and matched three-model results; current causal section records the pre-baseline access failure explicitly.
- figures: geometry full-depth curves and partial 8B/3.3-70B ledger figures embedded; causal transport and matched three-model figures remain unavailable.
- compile check: official January 2026 `neurips_2026.sty` is vendored from the public NeurIPS 2026 style source; `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` completed successfully (5 pages, no unresolved citations on final pass).

## CURRENT BEST DEFENSIBLE CLAIM
- On the 288-item Llama-3.1-8B layer-24 development artifact, equal-N mixed-format training AUC `0.6684` exceeds benchmark-only `0.6113` and casual-only `0.6259`, while pooled training is `0.7029`; the gap is not sample-size-only on this artifact. Across raw all-layer geometry, `norm(C)/norm(A)` versus benchmark→casual AUC correlates `-0.3814`, but this is one-model trajectory evidence, not independent-N significance or causal evidence.

## INPUT SERIALIZATION
- The canonical UTF-8/LF bank is frozen at `runs/tae_2026/input/rendered_items_canonical_lf.jsonl`, SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`. The deployed legacy `rendered_items.jsonl` has identical parsed rows/order but CRLF serialization (`4780e19e...`); new parent-owned jobs use the canonical bank. User-owned jobs were not modified; their resulting archive must pass exact semantic and byte-level provenance checks before inclusion.

## NEXT BLOCKER
- Validate the user-owned 3.1-70B archive or complete the canonical retry, then await parent-owned 3.3 analyses; the 8B causal retry remains blocked by model authorization, while the exact cached 3.3 causal path is active.
