# All-layer factorial geometry with fixed-dimensional controls

Run ID: `geometry_20260815_8b_all_layers_v2_controls`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `8cf6d2fd6506db7e2076d52135ceb7dd321e7742b40a863edfc564b2d9fb3b3b`  
Model: Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`; 32 layers, hidden size 4096; BF16 inference / float32 stored; final prompt token; zero-based block output before final norm  
Input SHA256: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`

## Controls

Every layer has three fixed 256-D random projections (seeds 1729, 2718, 31415) and a 256-D PCA control. Each PCA reducer is fitted using only the training-family rows in its held-out-family fold. No outcome-based seed or dimension selection is used.

Across layers, raw interaction ratio correlates with PCA ratio at `0.9993` and with random-projection-seed-1729 ratio at `0.9977`. PCA ratio ranges `0.2349–0.8224`; random-projection-seed-1729 ratio ranges `0.2763–0.8573`. The ratio-vs-benchmark→casual-AUC correlation is `-0.3662` for PCA and `-0.3795` for random projection, compared with `-0.3814` in raw space.

## Interpretation

The direction and approximate magnitude of the interaction-ratio trajectory are not a 4096-dimensional-only artifact in this model/development bank. The negative association with cross-format AUC persists under fixed-dimensional controls. Layers remain repeated measurements from one model, and these correlations are descriptive rather than independent-N inferential statistics. The B-projection calibration relationship remains weak in the current raw trajectory; no threshold mechanism is forced.

Full-depth output is machine-readable in `results.json`. The existing six geometry figures use the raw trajectory; fixed-dimensional figure overlays are still a presentation task. No causal outcome was used for layer selection; the frozen protocol selects shared layer 5 and interaction layer 16 from geometry-only eligibility.
