# Hostile review: final 8B mechanism claim

## Scope

Reviewed the frozen natural-target result, runner, executed and wrapper protocols, geometry selection protocol, activation provenance, baseline gate, and source item coverage. Primary claim is restricted to continued-forward internal transport toward the natural matched purpose counterfactual at downstream L24.

## Findings

### A/B/C fitting and target leakage — no critical issue

`results/run_causal_natural_counterfactual_controls.py:75-83` fits component means and strength statistics after excluding the held-out family. Held-out rows are used only to construct intervention deltas and evaluate target distances. Target indices are constructed at lines 166-176 after fitting; the natural target is same payload block, opposite purpose, same format. No target activation enters the direction fit.

### Layer-selection circularity — no issue

`runs/tae_2026/causal/causal_20260815_geometry_frozen/protocol.json` freezes eligibility and selection before causal inference: L5 is minimum eligible interaction ratio and L16 maximum. The causal result does not select layers from intervention outcomes.

### Correct/wrong-sign C — no issue

`make_deltas` implements full purpose transport as `-2*p*(A+f*C)`. Wrong-sign transport is `full_raw + 4*p*f*C = -2*p*(A-f*C)`, with strength fit on the corresponding training direction. Format-stratified L16 results are positive for correct C in benchmark (+0.004343) and casual (+0.009921), and negative for wrong C in benchmark (-0.009797) and casual (-0.012248).

### BF16 independence and endpoint boundary — no issue

The final natural-target run uses BF16 model inference and float32 stored representations. The quantized run is not used for the claim. The independent direct-logit purpose endpoint is near chance (accuracy/AUC 0.4757/0.4747), below the frozen gate; no behavioral steering claim is made.

### Random and norm controls — no critical issue

Random C vectors are Gaussian, orthogonalized to A, rescaled to `||C||`, and use the same training-projection-SD strength convention. The fixed 32-seed bank was selected before outcomes. Norm-matched A-only and coordinate-shuffled C are explicit controls.

### Family heterogeneity — limitation, not a fatal issue

Four of five family-level L16-minus-L5 contrasts are positive; ff03 is negative (-0.006291). Leave-one-family-out estimates remain positive, ranging from +0.006057 to +0.011015. The result is not driven by one family, but universal family-level sign consistency is not claimed. Exact two-sided five-family sign-test p is 0.375.

### Same-layer readout ordering — noncritical excluded artifact

`capture_batches` records downstream hidden states before applying the target-layer hook. Therefore intervention-layer/readout-layer L16/L16 records are unedited and are excluded from the frozen bundle. All primary claims use downstream L24, where the hook has propagated. No rerun is required for the primary endpoint because the affected same-layer readout is not used; the exclusion is recorded in `provenance.json` and `report.md`.

### Protocol binding — metadata defect fixed in consolidation

The historical batch passed the generic executed protocol, whose hash is recorded separately from the later natural-control wrapper protocol. The runner validated model, revision, and input SHA but did not validate natural-specific mode declarations. This is a reproducibility metadata defect, not a numerical-path defect. The frozen bundle explicitly records both executed and wrapper protocol paths/hashes, the runner hash, and the validation limitation; it does not claim that the wrapper protocol was the batch's original command-line protocol.

### Generic displacement, readout circularity, and metric artifact — limitations retained

The target-distance endpoint is an internal representation metric, not behavior. Selectivity is reported against the wrong-format natural target, and the direct-logit endpoint is excluded. The result remains vulnerable to broader questions about residual-stream bypass/path-specific mediation and downstream metric choice; it is therefore described as causal contribution to internal transport, not isolated path mediation or behavioral causation.

### Format balance and family coverage — no issue

The bank contains 72 complete 2x2 payload blocks across five families: ff01 has 16 blocks; ff02–ff05 have 14 each. Purpose and format cells are balanced at 144/144. No family or block leakage was found.

## Verdict

No critical numerical bug invalidates the bounded claim. The publishable claim is: correctly signed C improves internal natural-counterfactual transport at a layer independently selected as interaction-heavy, with no corresponding L5 benefit, and the effect survives wrong-sign, norm-matched, shuffled, and fixed-random controls. Do not call this behavioral steering, spontaneous evaluation awareness, or path-isolated mediation.
