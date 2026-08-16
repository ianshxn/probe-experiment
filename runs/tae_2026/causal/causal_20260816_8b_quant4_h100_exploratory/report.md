# Exploratory quantized 8B endpoint and internal readout

Evidence class: `EXPLORATORY_QUANTIZED_CAUSAL` for engineering/internal readouts only. This does not replace the frozen BF16 result.

## Baseline endpoint

Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
Quantization: bitsandbytes NF4 4-bit, BF16 compute.

The H100 run (`51554486`) completed in 49 seconds:

| endpoint | accuracy | AUC |
|---|---:|---:|
| purpose | 0.482639 | 0.491344 |
| format | 0.319444 | 0.349995 |

The independent T4 run (`51549381`) also completed, with purpose accuracy/AUC `0.486111/0.488088` and format accuracy/AUC `0.319444/0.352720`.

The frozen purpose endpoint threshold is `0.55`; therefore no direct-logit behavioral transport was run. Quantization does not rescue the endpoint gate.

## Internal exploratory transport

Job `51555587` completed in 54 seconds on one H100. It used the frozen L5/L16 layers, held-out purpose families, raw-space A/B/C directions fit separately within each quantized-model fold, continued the model forward, and evaluated downstream purpose projections at L24. Modes were zero, purpose full, purpose A-only, +/-A, and +/-C.

Projection scores are signed toward the original source purpose; lower values after a purpose flip indicate movement toward the counterfactual target.

| intervention layer | zero | full A+C | A-only | full-minus-A-only |
|---|---:|---:|---:|---:|
| L5 | 0.092028 | 0.092373 | 0.091378 | `+0.000995` |
| L16 | 0.092028 | 0.015397 | 0.028416 | `-0.013019` |

At L16, full A+C moves the downstream purpose projection an additional `0.013019` toward the counterfactual relative to A-only. L5 shows no corresponding interaction benefit. This is exploratory quantized internal evidence only; the endpoint gate prevents a behavioral causal interpretation.

Artifacts:

- baseline: `baseline.json`
- internal protocol: `../causal_20260816_8b_quant4_h100_exploratory_internal/protocol.json`
- internal results: `../causal_20260816_8b_quant4_h100_exploratory_internal/results.json`
