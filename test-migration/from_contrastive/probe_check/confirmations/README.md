# Prospective confirmations

These folders preserve each confirmation as one self-contained workflow. The
numbering is chronological and should not be read as escalating evidential
strength: each run tested a different frozen attempt to separate purpose from
lexical cue.

| confirmation | frozen approach | binding result |
|---|---|---|
| [01 — format matched](01_format_matched/README.md) | explicit-only probe, mean layer 6, fresh full-format controls | Rejected: purpose ordering passed, all eight cue-effect bounds failed. |
| [02 — cue invariant](02_cue_invariant/README.md) | SVD90 nuisance projection selected on spent bank 1, tested on fresh bank 2 | Rejected: purpose ordering passed, explicit-lexeme bounds failed in both languages on both models. |
| [03 — cue constrained](03_cue_constrained/README.md) | balanced-penalty-256 estimator selected on spent development controls, tested on bank 3 | Rejected: purpose ordering passed, Meta English implicit-cue effect exceeded the bound. |

All three banks are spent. The reports and metric files remain the authoritative
records; the notebooks and scripts are retained for reproducibility.

Historical metric JSON may contain the filenames that were current when a run
was executed. Those embedded provenance strings are intentionally unchanged;
the executable files now live in the numbered folders above.
