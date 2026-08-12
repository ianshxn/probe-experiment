# Confirmation 1: format-matched controls

This was the first prospective test of the selected explicit-only, mean-layer-6
probe against fresh controls rendered in the fitting format.

Execution order:

1. `build_format_matched_controls.py` creates the stable control bank at
   `probe_check/format_matched_controls.jsonl`.
2. `colab_extract_confirmatory_8b.ipynb` runs
   `extract_confirmatory_8b.py` on the two 8B stand-ins.
3. `evaluate_confirmatory_controls.py` writes the authoritative metrics to
   `probe_check/metrics/format_matched_confirmatory_controls.json`.

The frozen run instructions and final verdict are recorded in
[CONFIRMATORY_CONTROL_EXTRACTION_HANDOFF.md](CONFIRMATORY_CONTROL_EXTRACTION_HANDOFF.md).
It was rejected because all eight family-language lexical-cue bounds failed,
despite purpose ordering passing.
