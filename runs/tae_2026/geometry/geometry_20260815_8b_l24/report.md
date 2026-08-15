# Primary factorial geometry diagnostic

Run ID: `geometry_20260815_8b_l24`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `4fd6edd8e118a79b4765a9fd1ed9b32977d01fc925bc7126ccd0a8733f3766b7`  
Model: Llama-3.1-8B-Instruct, revision `0e9e39f249a16976918f6564b8830bc894c89659`; layer 24; BF16 inference / float32 storage; last prompt token; pre-final-norm zero-based block output  
Input: 288 rows, 72 complete blocks, five purpose families; input SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`  
Split: leave-one-purpose-family-out; factorial cells reconstructed by payload block and exact `(purpose, format)` key

## Estimand

For each block, the implementation computes exactly

`delta_B = h_EB - h_DB`, `delta_C = h_EC - h_DC`,

`A = (delta_B + delta_C)/4`, `C = (delta_B - delta_C)/4`, and
`B = ((h_EB+h_DB)-(h_EC+h_DC))/4`.

Fold-level geometry learns benchmark, casual, and pooled purpose directions from training families and evaluates held-out-family component vectors. The code reports norms, interaction ratio, direction cosines, principal cosines, and projections of held-out `B` onto training-derived purpose directions. Logistic transfer metrics use the same grouped folds and train-only standardization.

## Layer-24 numeric result

Mean held-family summaries:

- `norm(A) = 0.42496`
- `norm(B) = 7.34307`
- `norm(C) = 0.35475`
- `norm(C)/norm(A) = 0.84283`
- mean delta cosine = `0.19992`
- benchmark/casual purpose-direction cosine = `0.25777`

Transfer metrics from the same artifact:

- pooled AUC / balanced accuracy: `0.7029 / 0.6597`
- benchmark → casual: `0.5546 / 0.5278`
- casual → benchmark: `0.5656 / 0.5000`

These numbers are descriptive at one layer. They do not establish that the interaction ratio predicts transfer loss until the all-layer curve, fixed-dimensional controls, uncertainty resampling, and predeclared summary tests are complete.

## Synthetic falsification

`tests/test_geometry.py` passes fixed synthetic A-only, B-only, C-only, A+B, A+C, and noise cases. The tests recover the intended qualitative component and keep null components small.

## Incomplete required controls

This primary diagnostic records `fixed_dimensional_controls.implemented_in_this_primary_run=false`. The 256-D random-projection and PCA controls, full all-layer trajectories, bootstrap intervals, and paper-ready plots remain open. No causal layer is selected from this result.
