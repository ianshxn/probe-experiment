# Exploratory quantized internal mediation

Status: `EXPLORATORY_QUANTIZED_INTERNAL_MEDIATION`; job `51555587`; one H100; bitsandbytes NF4 4-bit with BF16 compute.

This run is not BF16 evidence, not behavioral causal steering, and does not replace the frozen BF16 result. The direct purpose endpoint baseline failed the frozen `0.55` gate (H100 quantized purpose accuracy/AUC `0.482639/0.491344`), so only internal downstream projection readouts were examined.

Using the frozen L5/L16 layers and held-out purpose-family folds, with A/B/C directions fit on each quantized model's training activations:

- L5 downstream L24 purpose projection: zero `0.092028`, full A+C `0.092373`, A-only `0.091378`; full-minus-main `+0.000995`.
- L16 downstream L24 purpose projection: zero `0.092028`, full A+C `0.015397`, A-only `0.028416`; full-minus-main `-0.013019`.

Signed projections are oriented toward the original source purpose; a more negative value after a purpose flip indicates stronger movement toward the counterfactual. The exploratory quantized result therefore reproduces the qualitative layer-selective pattern: interaction inclusion matters at L16 but not L5. It remains an exploratory internal readout, not a behavioral claim.
