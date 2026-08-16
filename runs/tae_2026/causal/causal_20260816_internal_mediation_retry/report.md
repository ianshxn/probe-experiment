# Internal causal mediation: Llama-3.1-8B

Run ID: `causal_20260816_internal_mediation_retry`

Status: complete. This is **internal causal mediation**, not behavioral causal steering. The direct candidate-logit endpoint was validated first and failed the frozen gate: purpose accuracy/AUC `0.475694/0.474706`; format accuracy/AUC `0.569444/0.598827`. No candidate-token causal result is reported.

## Frozen design

- Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Input: canonical 288-item bank, SHA256 `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`.
- Held-out-family cross-fitting: five purpose families; every intervention is fit on the remaining families.
- Intervention layers: shared layer 5 and interaction-heavy layer 16, selected from geometry before this run.
- Downstream projection readouts: layers 16, 24, and 31. Primary summary below uses layer 24.
- Full factorial transport uses `-2p(A+fC)` for purpose and `-2f(B+pC)` for format. Main-effect-only transport removes `C`.
- Raw residual-space unit directions; strength is one training-block projection standard deviation.
- Modes include full/main transport, signed `+/-A`, `+/-B`, `+/-C`, zero-strength, sign controls, and 32 fixed random directions.
- The model forward continues after the residual-stream edit. Scores are float32.

The frozen protocol is bundled as `protocol.json`; raw output is `results.json` (860 held-family/readout/mode records).

## Results

Signed target projection is oriented so that a successful counterfactual flip decreases the held-out example's original-factor projection. Values below are mean changes from the matched zero-strength control over the five held-out families. Negative target changes therefore indicate movement toward the counterfactual factor.

| source direction | intervention layer | target readout | full transport | main-effect-only | full minus main |
|---|---:|---|---:|---:|---:|
| purpose | 5 | purpose, L24 | `+0.00019` | `-0.00140` | `+0.00159` |
| purpose | 16 | purpose, L24 | `-0.07959` | `-0.06665` | `-0.01294` |
| format | 5 | format, L24 | `-0.00059` | `-0.00033` | `-0.00026` |
| format | 16 | format, L24 | `-0.20042` | `-0.19853` | `-0.00189` |

The layer-16 effects are consistent across held-out families: purpose-full purpose changes were `[-0.0554, -0.0758, -0.0919, -0.0861, -0.0888]` for the purpose source and `[-0.0560, -0.0797, -0.1014, -0.0955, -0.0945]` for the format source. Format-full format changes were `[-0.1842, -0.2040, -0.2109, -0.1997, -0.2032]` for the format source.

The interaction term contributes materially to purpose transport at layer 16: full transport is approximately `0.013` more negative than main-effect-only transport for the purpose source. The corresponding format increment is small (`0.002`). Layer 5 is effectively null under the same readout, consistent with the geometry-based shared/interaction layer contrast.

## Selectivity and controls

At layer 16, purpose-full transport changed the off-target format projection by only `+0.00049` (purpose source) and `+0.00097` (format source). Format-full transport changed the off-target purpose projection by `+0.00004` and `-0.00120`, respectively. This supports internal factor selectivity in the downstream projection readouts.

The 32 random-direction controls were centered near zero. For the purpose target at layer 16, random mean change was `+0.000184`, SD `0.000632`, range `[-0.001801, +0.001653]`; the predeclared purpose-full effect was `-0.079593`. For the format target, random mean change was `+0.000660`, SD `0.001164`, range `[-0.002391, +0.003433]`; the predeclared format-full effect was `-0.200418`.

## Interpretation boundary

The frozen residual-stream transport produces strong, held-out-family, factor-selective **internal** mediation at layer 16, and including `C` improves purpose counterfactual movement over the main-effect-only ablation. This is evidence that the factorial interaction is mechanistically relevant to the internal representation.

It is not behavioral causal steering: the direct endpoint was near chance and was correctly excluded before inference. The result also does not establish that the model's generated behavior changes. The workshop claim should therefore be limited to training-derived residual transport and downstream internal readouts until an independently validated behavioral endpoint is available.
