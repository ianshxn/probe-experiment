# Independent 8B causal run after access approval

Run: `causal_20260816_internal_mediation_access_granted`
Job: `51536964`
GPU: one `H100-80GB`

## Access and baseline gate

The lightweight preflight fetched the exact revision metadata, `config.json`, `tokenizer_config.json`, and fast tokenizer using the approved cluster credential mechanism without printing a token. Resolved revision: `0e9e39f249a16976918f6564b8830bc894c89659`. Classification: `ACCESS_OK`.

The independent unsteered baseline job was `51535654` and completed in 15 seconds. It used the canonical 288-item bank, BF16 inference, the frozen candidate mapping seed `2026081533`, and the frozen wrappers. Candidate-token validation passed for both mappings.

| endpoint | wrapper | mapping | accuracy | AUC |
|---|---|---|---:|---:|
| purpose | `purpose_v1` | evaluation=B, deployment=A | 0.475694 | 0.474706 |
| format | `format_v1` | benchmark=B, casual=A | 0.569444 | 0.598814 |

Purpose endpoint metrics by held-out family (accuracy, AUC):

- `ff01`: `0.4375`, `0.455566`
- `ff02`: `0.517857`, `0.498724`
- `ff03`: `0.428571`, `0.460459`
- `ff04`: `0.517857`, `0.464923`
- `ff05`: `0.482143`, `0.470026`

Format endpoint metrics by held-out family (accuracy, AUC):

- `ff01`: `0.640625`, `0.708008`
- `ff02`: `0.357143`, `0.342474`
- `ff03`: `0.517857`, `0.628827`
- `ff04`: `0.678571`, `0.786990`
- `ff05`: `0.642857`, `0.573980`

Because purpose accuracy was below the frozen `0.55` endpoint threshold, behavioral direct-logit transport was **not run**. No held-out activation was modified for a behavioral causal claim. The gate failure is preserved in `causal_20260816_8b_transport_access_granted_gate_failed/failure.json`.

## Frozen internal causal fallback

The independent fallback job ran the already-frozen factorial residual-stream protocol without any 70B or comparison dependency. It continued the model forward after each intervention and produced 860 held-family/readout/mode records.

- Intervention layers: L5 and L16.
- Downstream projection readouts: L16, L24, L31.
- Full purpose: `-2*p*(A_hat + f*C_hat)`.
- Purpose A-only: `-2*p*A_hat`.
- Full format: `-2*f*(B_hat + p*C_hat)`.
- Format B-only: `-2*f*B_hat`.
- Modes: `+/-A`, `+/-B`, `+/-C`, zero-strength, and 32 fixed random directions.
- Strength: one training-block projection standard deviation in raw residual space.
- Fits: training purpose families only within each held-out-family fold.

Signed target projection changes below are relative to the matched zero-strength control and averaged over the five held-out families. Negative values indicate movement toward the counterfactual target under the readout orientation.

| source activation | intervention | target readout at L24 | full | main effect only | full − main |
|---|---|---|---:|---:|---:|
| purpose | L5 purpose flip | purpose | `+0.000190` | `-0.001395` | `+0.001585` |
| purpose | L16 purpose flip | purpose | `-0.079593` | `-0.066654` | `-0.012939` |
| format | L5 format flip | format | `-0.000588` | `-0.000326` | `-0.000262` |
| format | L16 format flip | format | `-0.200418` | `-0.198530` | `-0.001888` |

The pre-frozen primary prediction is supported internally: including `C` improves purpose counterfactual movement at L16 by about `0.0129` signed projection units, while L5 shows no comparable improvement. This is an internal readout result, not behavioral steering.

## Selectivity and controls

At L16, purpose-full transport from purpose activations changed the purpose projection by `-0.079593` and the off-target format projection by `+0.000490`. Format-full transport from format activations changed the format projection by `-0.200418` and the off-target purpose projection by `-0.001201`.

Thirty-two fixed random controls were centered near zero:

- purpose target: mean `+0.000184`, SD `0.000632`, range `[-0.001801, +0.001653]`;
- format target: mean `+0.000660`, SD `0.001164`, range `[-0.002391, +0.003433]`.

The signed component controls were also executed. At L16, `+B/-B` moved the format projection (`+0.021615/-0.028571` for purpose-source activations; `+0.015117/-0.022266` for format-source activations). Simple `+/-A` and `+/-C` component effects were small and variable relative to the full factorial transport; they are retained as null/weak controls rather than upgraded to a positive claim.

## Boundary

This run establishes an independently reproduced, held-out-family internal mediation result after the exact 8B access gate passed. It does not establish generated-behavior change: the direct purpose endpoint failed its predeclared baseline gate, so behavioral transport was correctly refused.
