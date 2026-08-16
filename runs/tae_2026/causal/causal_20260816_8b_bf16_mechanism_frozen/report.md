# Frozen 8B BF16 mechanism bundle

## Claim boundary

The BF16 Llama-3.1-8B result tests internal causal counterfactual transport under a controlled prompt-provided purpose×format variable. The direct purpose-logit endpoint failed its frozen gate; no behavioral steering or spontaneous evaluation-awareness claim is made. All reported causal distances use downstream L24; the same-layer L16/L16 records are excluded because the runner records the readout before applying the target-layer hook.

## Family robustness

| family | payloads | L5 C benefit | L16 C benefit | L16−L5 |
|---|---:|---:|---:|---:|
| ff01 | 16 | -0.000855 | +0.009526 | +0.010380 |
| ff02 | 14 | -0.000718 | +0.013530 | +0.014248 |
| ff03 | 14 | +0.003216 | -0.003075 | -0.006291 |
| ff04 | 14 | -0.001361 | +0.007967 | +0.009328 |
| ff05 | 14 | -0.002825 | +0.007370 | +0.010195 |

4/5 family contrasts are positive; ff03 is negative. Exact two-sided sign-test p=0.3750; exact sign-flip one-sided finite-family statistic=0.0909. Leave-one-family-out range: [+0.006057, +0.011015].

## Effect sizes

Fraction of natural source-target gap closed by C: L5 mean -0.000016, L16 mean +0.004166, paired L16−L5 +0.004182. Bootstrap intervals are in `effect_sizes.json`.
Random-bank descriptive standardized effect: +6.758; finite-bank empirical statistic 0.0303 = (1+0)/(1+32). This is not a Gaussian z-test or population-level p-value.

## Wrong-sign by format

The format-stratified table is in `wrong_sign_by_format.json`; it reports correct-sign and wrong-sign C increments relative to A-only for benchmark and casual source cells at L5 and L16.

## Selectivity

Selectivity is defined once as improvement toward the correct purpose target minus improvement toward the wrong-format target. All components and family rows are in `selectivity.json`.

## Provenance

Frozen from commit `1b73654aade70c29291802eb74c17c1a80923837`; source hashes and fit scope are in `provenance.json`.
