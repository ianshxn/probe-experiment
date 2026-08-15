# Geometry conditional family bootstrap

Run ID: `geometry_20260815_8b_all_layers_v3_bootstrap`  
Input: `geometry_20260815_8b_all_layers_v2_controls/results.json`  
Output SHA256: `19f7905ca6981915a22124c6c5f53a22070df1d49dd700aa3786e936e82044d8`  
Bootstrap: 1,000 resamples per layer, seed `20260815`, unit `purpose_family_id`.

This artifact adds per-layer 95% bands for norm(A), norm(B), norm(C), norm(C)/norm(A), matched delta cosine, and benchmark/casual purpose-direction cosine. The bands resample held-out-family summaries conditional on the already fitted fold models; they do not refit probes. They are uncertainty bands for family generalization, not independent layer-level inferential tests. The source all-layer raw/PCA/random-projection result remains immutable.
