# Causal protocol gate

Protocol SHA256: `d651da40a535526d382936a42428aa5ce0e816d1ff20cba8b79f55dadec140c9`  
Baseline artifact: `baseline.json`  
Status: **blocked before steering**

Geometry-only layer selection is frozen: shared layer 5 (lowest interaction ratio among eligible layers) and interaction layer 16 (highest interaction ratio among eligible layers), with eligibility pooled purpose AUC >= 0.80 and exclusion of first/last layer. No causal outcome was used for selection.

The protocol freezes raw-space A/B/C transport, full versus main-effect-only ablations, endpoint wording, randomized candidate mapping, training-derived strength, signs, zero strength, and 32 random controls. The local environment has torch/transformers packages but no usable local model endpoint; the Llama-3.1-70B BF16 job 51479700 is pending. Baseline endpoint accuracy and candidate-token validation therefore remain unrun. No steering result is reported, and no causal claim is made.
