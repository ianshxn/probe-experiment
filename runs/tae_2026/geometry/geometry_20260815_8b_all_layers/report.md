# All-layer factorial geometry (development diagnostic)

Run ID: `geometry_20260815_8b_all_layers`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `7f3c344292c3f3909c85c8d84da2bb048a3a8ea8ce5456a031dfc11399b996e3`  
Model: Llama-3.1-8B-Instruct, revision `0e9e39f249a16976918f6564b8830bc894c89659`; 32 layers; hidden size 4096; BF16 inference / float32 stored; final prompt token; pre-final-norm zero-based block output  
Input SHA256: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`

## Full-trajectory numeric inspection

Across layers, `norm(C)/norm(A)` ranges from `0.2780` to `0.8428`; benchmark→casual AUC ranges from `0.5235` to `0.8451`. The Pearson correlation across the 32 layer trajectory points is `-0.3814` between interaction ratio and cross-format AUC, and `-0.3204` between interaction ratio and cross-format balanced accuracy. This is directionally consistent with the predeclared hypothesis but is not an independent-N significance test: layers are repeated measurements from one model.

The best benchmark→casual AUC is `0.8451` at layer 8; the worst is `0.5235` at layer 2. These are trajectory extrema, not a basis for a best-layer headline. The B-projection versus AUC-minus-balanced-accuracy calibration-gap correlation is `0.2286` for the benchmark-trained direction. The calibration relationship is therefore weak in this first pass and should not be forced.

## Outputs

Full curves are in `runs/tae_2026/figures/geometry_20260815_8b_all_layers/`:

1. interaction ratio vs depth;
2. cross-format AUC vs depth;
3. interaction ratio vs cross-format AUC;
4. B projection vs AUC-minus-balanced-accuracy gap;
5. delta cosine vs depth;
6. principal-subspace overlap vs depth.

The primary-layer fixed-dimensional control artifact is `runs/tae_2026/geometry/geometry_20260815_8b_l24_v2_controls/`. Full-depth fixed-dimensional curves and family/bootstrap bands remain open. No causal layer is frozen from this exploratory trajectory yet.
