# Cross-fitted representational counterfactual reconstruction

Status: `DEVELOPMENT_OFFLINE_MECHANISM`.

This is an observational activation-space reconstruction. It does not continue the model forward and is not behavioral causality or causal steering.

- L5 purpose C benefit: 0.000107 (DB->EB); mean four-purpose-direction benefit 0.000402.
- L16 purpose C benefit: 0.032771 (DB->EB); mean four-purpose-direction benefit 0.005555.
- Paired L16-minus-L5 bootstrap contrast: 0.005153; 95% CI [-0.000347, 0.010752].
- All-layer ratio/benefit association: Pearson -0.1213; Spearman -0.0836.

Raw per-payload metrics, PCA controls, random-projection controls, provenance, and fixed seeds are in `results.json`.

PCA and random-projection controls are included in the final `results.json`; PCA is fit separately within each held-out-family training fold and fixed random projections use seeds 1729, 2718, and 31415.
