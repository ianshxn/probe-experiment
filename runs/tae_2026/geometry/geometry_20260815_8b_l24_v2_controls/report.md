# Factorial geometry with fixed-dimensional controls

Run ID: `geometry_20260815_8b_l24_v2_controls`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `f9dbf09f5a8e4c68191fdc7a7be1b141a1c6bf246ac9f4341b8d758abb70a696`  
Model/revision: Llama-3.1-8B-Instruct / `0e9e39f249a16976918f6564b8830bc894c89659`  
Input: 288 rows / 72 complete payload blocks / five families; item SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`  
Layer: 24, zero-based transformer-block output before final norm; last prompt token; BF16 inference / float32 stored

## Raw-space geometry

Mean held-family values are `norm(A)=0.42496`, `norm(B)=7.34307`, `norm(C)=0.35475`, `norm(C)/norm(A)=0.84283`, delta cosine `0.19992`, and benchmark/casual purpose-direction cosine `0.25777`. Transfer is pooled AUC/BA `0.7029/0.6597`, benchmark→casual `0.5546/0.5278`, and casual→benchmark `0.5656/0.5000`.

## Dimensionality controls

The fixed dimension is 256. Random projections use seeds 1729, 2718, and 31415 and are generated independently of outcomes. Their interaction ratios are `0.83564`, `0.86664`, and `0.85697`; their delta cosines are `0.21094`, `0.16950`, and `0.17998`. The fixed 256-D PCA control fits each reducer on training-family rows within each held-family fold; its summary is `norm(A)=0.32389`, `norm(B)=7.24788`, `norm(C)=0.26277`, `norm(C)/norm(A)=0.82243`, delta cosine `0.22131`, and purpose-direction cosine `0.25777`.

The interaction ratio is qualitatively stable in these primary-layer controls. This is not yet the required all-layer predictive test: no correlation across depth, bootstrap curve, or predeclared causal layer has been produced. The artifact remains development evidence and makes no causal claim.

## Falsification and limitations

`tests/test_geometry.py` passes the fixed A-only, B-only, C-only, A+B, A+C, and null cases. Structural cell reconstruction rejects duplicate or incomplete blocks. The current 288-item artifact has uneven coverage and lacks an explanation for the missing eight intended blocks; this remains a provenance limitation.
