# Final BF16 internal causal replication

## Evidence boundary

This is a held-out-purpose-family residual-stream intervention result. The model is `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`, on the frozen 288-item bank. Inference used BF16 and continued the forward pass. The frozen direct candidate-logit endpoint failed its purpose gate, so no behavioral steering claim is made.

## Projection readout

Purpose projection at downstream L24, averaged over five held-out families:

| intervention layer | full A+fC | A-only | full minus A-only |
|---:|---:|---:|---:|
| L5 | 0.096791 | 0.095206 | +0.001585 |
| L16 | 0.017007 | 0.029946 | -0.012939 |

The paired L16-minus-L5 contrast of the full-minus-A-only effect is `-0.014524`. Complete held-out-family bootstrap over the five family means (100,000 fixed-seed replicates, seed `2026081602`) gives 95% CI `[-0.017485, -0.012023]`; all bootstrap draws are negative. This exact BF16 replication therefore preserves the layer-selective interaction effect, with the sign convention that lower downstream purpose-projection error/movement is closer to the natural counterfactual in the companion natural-target analysis.

The final run contains 860 held-out records across intervention/readout layers and frozen controls. Its direct endpoint status remains rejected by the predeclared purpose accuracy/AUC gate.

## Natural downstream counterfactual target

The companion run `causal_20260816_8b_bf16_natural_counterfactual_controls` compares each intervened held-out activation to the matched natural purpose-counterfactual activation after continued forward propagation. At intervention L16 and readout L24:

| mode | mean raw distance to natural target | mean purpose-target improvement relative to zero |
|---|---:|---:|
| zero | 1.850649 | 0.000000 |
| full A+fC | 1.833561 | +0.017088 |
| A-only | 1.840693 | +0.009956 |
| norm-matched A-only | 1.837914 | +0.012736 |
| wrong-sign C | 1.851715 | -0.001066 |
| shuffled C | 1.843236 | +0.007414 |

The incremental full-minus-A-only benefit, expressed as reduction in target distance, is `+0.007132` at L16 versus `-0.000518` at L5; the paired L16-minus-L5 contrast is `+0.007650`. Bootstrap over 288 complete payload cells gives 95% CI `[+0.002978, +0.012349]`; bootstrap over 72 payload blocks gives `[+0.002189, +0.013014]`; family-mean bootstrap gives `[+0.000303, +0.012453]`.

At L16, wrong-sign C gives an incremental effect of `-0.011023` versus A-only; norm-matched A-only gives `+0.002779`; coordinate-shuffled C gives `-0.002543`. The 32 fixed random orthogonal-C controls have L16-minus-L5 incremental-benefit mean `-0.001825`, SD `0.001402`, 2.5/50/97.5 percentiles `[-0.003850, -0.002384, +0.001016]`; the observed `+0.007650` is beyond all 32 controls (empirical one-sided p `0/32`).

At L5, full transport does not improve natural-target distance over A-only (`-0.000518`), matching the frozen low-interaction-layer prediction. At L16, correctly signed C improves target transport while wrong-sign, shuffled, and random controls do not reproduce the effect. This supports an internal causal mediation interpretation of the factorial interaction; it does not establish a behavioral endpoint effect.
