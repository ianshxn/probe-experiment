# Equal-N and target-format exposure control

Run ID: `equal_n_20260815_8b_l24_v3_target_exposure`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `460548e4288855ed6665f703e85100be445fedc5114b8926b82f634d27f1efbf`  
Model: Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`; layer 24; BF16 inference / float32 stored; final prompt token; pre-final-norm block output  
Input SHA256: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`

The validated equal-N design and structural checks are unchanged. This immutable run adds same-format reference tests to separate target-format exposure from cross-format transport. Means over the fixed seed bank:

| regime | AUC | balanced accuracy |
|---|---:|---:|
| benchmark-only → benchmark (same format) | 0.7444 | 0.6875 |
| benchmark-only → casual (cross format) | 0.5544 | 0.5208 |
| casual-only → casual (same format) | 0.7095 | 0.6181 |
| casual-only → benchmark (cross format) | 0.5656 | 0.5000 |
| equal-N mixed-format → all formats | 0.6684 | 0.6250 |
| pooled/decorrelated → all formats | 0.7029 | 0.6597 |

Same-format performance is substantially higher than cross-format performance for both one-format training regimes. Equal-N mixed-format training improves over one-format cross-format transfer while using the same number of training rows as each one-format regime. This supports a target-format exposure contribution to the cross-format gap, while leaving the shared/interaction geometry as the mechanistic question. It is not a causal result.

The artifact includes five fixed assignment seeds, structural validation, per-family metrics, and conditional family bootstrap intervals. Full-depth target-exposure curves are still pending.
