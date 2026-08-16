# 8B causal transport endpoint gate

Run ID: `causal_20260816_8b_transport_retry_failed`

## Status

The exact Llama-3.1-8B model loaded successfully after HF access approval, and the frozen direct-logit baseline executed on all 288 canonical items. The held-out transport job then refused to run because the frozen endpoint gate detected near-chance purpose readout performance.

- Baseline job: `51515708`
- Transport job: `51515710`
- Model revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- Input SHA256: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`
- Baseline artifact SHA256: `d92c3f481aeafb8c0cd9defa77da4441b38ca29bfe4942309dba677f7d32b5fe`

## Frozen endpoint results

| Readout | Accuracy | AUC |
|---|---:|---:|
| Purpose | 0.4757 | 0.4747 |
| Format | 0.5694 | 0.5988 |

The transport runner raised `RuntimeError: baseline endpoint is near chance; refusing causal transport`. This is an endpoint-validation failure, not a null intervention result. No held-out activation was modified, no transport outcome exists, and no causal claim is made.

The frozen protocol explicitly requires baseline validation before steering and rejects endpoint accuracy below 0.55. The purpose endpoint is near chance under the predeclared randomized mapping seed `2026081533`; the result is preserved rather than replaced with an outcome-dependent wrapper or mapping.
