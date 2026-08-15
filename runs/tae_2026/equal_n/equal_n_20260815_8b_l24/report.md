# Equal-N mixed-format control (development diagnostic)

Run ID: `equal_n_20260815_8b_l24`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `c12d472ad47c4d7261c267e82cb0568e1af4c6eb064400ac323f508ec883f16e`  
Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`  
Layer: 24, zero-based transformer-block output before final model norm; final prompt token; BF16 inference / float32 stored  
Input: 288 rows, 72 payload blocks, five purpose families; item SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`  
Regularization: logistic regression `C=0.1`; split: leave-one-purpose-family-out; pairs and blocks retained

## Design check

The fixed seed bank is `[1729, 2718, 31415, 4242, 8675309]`. For every held-out family, benchmark-only, casual-only, and equal-N mixed-format training each contain exactly two rows per training payload block. Equal-N assigns one format to each block and keeps its evaluation/deployment pair together; benchmark/casual assignment differs by at most one block. The result records `equal_training_rows_first_three: true` and the pooled regime has exactly twice as many training rows.

## Primary layer-24 results

Metrics below are means over the five fixed balanced assignments. AUC and threshold balanced accuracy are separate quantities.

| training regime | AUC | balanced accuracy |
|---|---:|---:|
| benchmark-only, all-format test | 0.6113 | 0.6042 |
| casual-only, all-format test | 0.6259 | 0.5590 |
| equal-N mixed-format, all-format test | **0.6684** | **0.6250** |
| pooled/decorrelated, all-format test | 0.7029 | 0.6597 |
| benchmark → casual | 0.5544 | 0.5208 |
| casual → benchmark | 0.5656 | 0.5000 |

Equal-N mixed-format AUC standard deviation across assignments is 0.0041 and balanced-accuracy standard deviation is 0.0163. The benchmark-only, casual-only, pooled, and one-format transfer scores are assignment-invariant because their training selectors do not use the seed bank.

## Interpretation

On this development bank and layer, the original pooled-versus-one-format gap does **not** disappear after matching training row count. Equal-N mixed-format training remains above benchmark-only and casual-only training in AUC (0.6684 versus 0.6113/0.6259), while pooled training remains higher (0.7029). This is evidence that the unequal-N confound is not sufficient to explain the observed gap, not evidence by itself for a format-conditioned representation: the equal-N condition still exposes training blocks to both formats, and the one-format transfer conditions do not.

The appropriate next comparison is factorial geometry plus target-format-exposure controls. AUC and balanced accuracy diverge, especially for casual-only and casual→benchmark, so threshold calibration must not be collapsed into ranking performance. This result is exploratory/development evidence until block/family uncertainty, structural validation, dimensionality controls, and matched model replication are complete.
