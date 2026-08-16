# BF16 natural-target counterfactual controls

The frozen direct-logit endpoint failed its purpose gate; this run is internal causal mediation only. It uses the exact 288-item bank, Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`, BF16 inference, held-out purpose-family fits, and continued forward passes.

At intervention L16 and downstream readout L24, mean raw Euclidean distance to the natural matched purpose-counterfactual target was 1.850649 for zero, 1.833561 for full $A+fC$, 1.840693 for $A$-only, 1.837914 for norm-matched $A$-only, 1.851715 for wrong-sign $C$, and 1.843236 for shuffled $C$. The full-minus-$A$-only reduction in target distance was +0.007132 at L16 and -0.000518 at L5. The paired L16-minus-L5 benefit was +0.007650; complete-payload bootstrap 95% CI `[+0.002978,+0.012349]`, payload-block CI `[+0.002189,+0.013014]`, and family-mean CI `[+0.000303,+0.012453]`.

Wrong-sign $C$ produced an L16 incremental effect of -0.011023 relative to $A$-only; norm-matched $A$-only produced +0.002779; shuffled $C$ produced -0.002543. The 32 fixed random orthogonal-$C$ controls had L16-minus-L5 incremental-benefit mean -0.001825, SD 0.001402, 2.5/50/97.5 percentiles `[-0.003850,-0.002384,+0.001016]`; none exceeded the observed +0.007650.

The result is consistent with correctly signed interaction transport improving internal movement toward the natural counterfactual at L16, while the low-interaction L5 control shows no benefit. It does not establish behavioral steering.
