# Confirmation 3: constrained probe

This workflow froze the `balanced_penalty_256` estimator after cue-balanced
development and tested it once on the fresh third control bank.

Execution order:

1. `build_cue_constrained_confirmatory_controls.py` creates the third bank at
   `probe_check/cue_constrained_confirmatory_controls.jsonl`.
2. `colab_extract_cue_constrained_confirmatory_8b.ipynb` runs
   `extract_cue_constrained_confirmatory_8b.py` on the two 8B stand-ins.
3. `evaluate_cue_constrained_confirmation.py` performs the frozen evaluation.

See [the handoff](CUE_CONSTRAINED_CONFIRMATION_HANDOFF.md) and
[binding result](RESULT_cue_constrained_confirmation_8b.md). Every purpose gate
passed, but Meta English implicit cues measured 0.3566 fitting SD against the
0.25 maximum, so the recipe was rejected.
