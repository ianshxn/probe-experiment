# Equal-N mixed-format control (validated development diagnostic)

Run ID: `equal_n_20260815_8b_l24_v2_validated`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `2308584a7ebd2f953b226f7bb84027d41196a0c379baa8e1e0009e8fae30a4d7`  
Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`; layer 24; BF16 inference / float32 stored; final prompt token; pre-final-norm zero-based block output  
Input: 288 rows, 72 blocks, five families; item SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`  
Regularization: `C=0.1`; fixed assignment seeds `[1729, 2718, 31415, 4242, 8675309]`

## Structural and training-size checks

The implementation rejects duplicate item IDs, incomplete 2×2 payload blocks, multiple families per block, split-group mismatches, and label/purpose mismatches. Every held-out family keeps complete blocks untouched. Benchmark-only, casual-only, and equal-N mixed training have identical row counts in every fold; pooled training has twice the rows. Equal-N assignments are balanced to within one block.

## Primary layer-24 results

Means over the fixed five assignment seeds; AUC and balanced accuracy are separate:

| regime | AUC | balanced accuracy |
|---|---:|---:|
| benchmark-only, all-format test | 0.6113 | 0.6042 |
| casual-only, all-format test | 0.6259 | 0.5590 |
| equal-N mixed-format, all-format test | **0.6684** | **0.6250** |
| pooled/decorrelated, all-format test | 0.7029 | 0.6597 |
| benchmark → casual | 0.5544 | 0.5208 |
| casual → benchmark | 0.5656 | 0.5000 |

Equal-N assignment standard deviations are 0.0041 AUC and 0.0163 balanced accuracy. Conditional family bootstrap (1,000 resamples; fixed fold fits; unit `purpose_family_id`) gives equal-N mixed AUC 95% CI `[0.6262, 0.7345]` and balanced-accuracy 95% CI `[0.5910, 0.6577]`. These intervals quantify family variation conditional on the fitted outer-fold models; they are not a refit bootstrap.

## Interpretation

The pooled-versus-one-format gap does not disappear after matching training row count on this development artifact. Equal-N mixed-format AUC remains above benchmark-only and casual-only, while pooled training remains higher. This rejects a sample-count-only explanation for this bank, but it does not isolate target-format exposure or establish a format-conditioned representation. The 288-item bank also has 72 blocks rather than the intended 80 and uneven family/topic coverage; that provenance limitation remains open.

The AUC/threshold split is material and is preserved as separate outcomes. Full-depth geometry, fixed-dimensional controls, target-format exposure controls, and model replication are required before a workshop claim.
