# Hostile reproducibility review

Run ID: `hostile_20260815_initial`  
Git commit at sprint start: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Status: **critical findings remain open**

## Critical findings

1. The original pooled/decorrelated versus one-format comparison had unequal training rows and target-format exposure. The validated equal-N layer-24 run removes row-count equality as an explanation on this bank, but target-format exposure remains a distinct control.
2. The checked-in activation bank has 288 rows / 72 blocks, while repository design notes describe an intended 320 / 80 grid. Family and topic coverage are uneven and the missing-block provenance is not documented.
3. Equal-N and geometry now reject duplicate/incomplete factorial blocks and label/purpose mismatches, but equal-N family bootstrap is conditional on fixed fold fits rather than a refit bootstrap.
4. Full-depth raw geometry is implemented; fixed 256-D random/PCA controls are complete for the primary layer but not the full curve.
5. The current analysis does not yet have a causal endpoint baseline. Protocol and layer selection are frozen, but steering is blocked before baseline because no local model endpoint is available and the exact 3.1-70B extraction is pending.
6. Existing legacy extraction paths have incompatible layer/token semantics and must not be mixed with the current zero-based last-token archive.
7. Current 3.3-70B local evidence has revision metadata in the extraction config but not in the prior sweep result; matched comparison provenance must be repaired.

## Checks performed

- purpose-family-held-out folds and complete payload-block grouping inspected;
- exact item order/hash and 8B activation dimensions verified;
- all-layer array shape N×L×D and layer IDs verified;
- equal-N training counts verified equal for benchmark-only/casual-only/mixed in every fold;
- synthetic A-only, B-only, C-only, A+B, A+C, and null geometry cases pass;
- causal hook, zero-strength, sign/target-token, and finite-logit focused tests pass;
- 3.1-70B cache absent before job submission; exact Hub revision resolved and BF16 all-layer job 51479700 submitted on approved paths.

## Status and fixes

- **Fixed:** equal-N control, deterministic seed bank, structural row checks, all-layer raw A/B/C decomposition, primary fixed-dimensional controls, frozen geometry-only causal protocol, focused synthetic/hook tests.
- **Open:** full-depth fixed-dimensional controls, refit uncertainty, target-format exposure decomposition, causal endpoint baseline/transport, 3.1-70B extraction, 72-versus-80 coverage explanation, final provenance lock.

No headline causal or three-model claim is currently defensible.
