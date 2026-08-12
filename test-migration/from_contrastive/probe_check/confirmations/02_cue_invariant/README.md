# Confirmation 2: nuisance-projected cue invariance

This workflow selected an SVD90 nuisance projection on the spent first bank and
then evaluated it once on a fresh, text-disjoint second bank.

Execution order:

1. `build_cue_invariant_controls.py` creates the second bank at
   `probe_check/cue_invariant_confirmatory_controls.jsonl`.
2. `develop_cue_invariant.py` selects the projection using spent development
   controls only.
3. `colab_extract_cue_invariant_confirmatory_8b.ipynb` runs
   `extract_cue_invariant_confirmatory_8b.py` on the two 8B stand-ins.
4. `evaluate_cue_invariant_confirmation.py` performs the frozen evaluation.

See [the handoff](CUE_INVARIANT_CONFIRMATION_HANDOFF.md),
[development result](RESULT_cue_invariant_development_8b.md), and
[binding confirmation](RESULT_cue_invariant_confirmation_8b.md). The recipe was
rejected because explicit-lexeme effects exceeded the bound in both languages
on both models, although all purpose gates passed.
