# Probe validation workspace

This directory tests whether a linear probe can recover evaluation-versus-
deployment purpose without merely learning lexical cues in the prompt. The
stand-in 8B runs are design validation, not evidence about the declared subject
models. Naturalistic transfer remains sealed.

## Layout

```text
probe_check/
├── confirmations/            # the three prospective confirmation workflows
│   ├── 01_format_matched/
│   ├── 02_cue_invariant/
│   └── 03_cue_constrained/
├── development/              # exploratory analyses that selected recipes
│   ├── baseline/
│   ├── layer_selection/
│   └── cue_constraints/
├── metrics/                  # compact numerical records
├── out/                      # local activations and logs; ignored by Git
├── analyze.py, fit.py        # shared historical/general evaluators
├── cue_*.py                  # shared probe estimators and data utilities
└── *.jsonl                   # stable generated control-bank assets
```

Start with the [confirmation index](confirmations/README.md) for the binding
results and the [development index](development/README.md) for the analyses
that led to them. The complete interpretation and redesign boundary are in
[the archived probe-redesign record](../docs/PROBE_REDESIGN_ARCHIVE_2026_08_06.md).

## Current conclusion

All three prospective 8B confirmations are complete and rejected under their
predeclared lexical-cue bounds. The latest constrained recipe preserved every
purpose-ordering gate but missed one fresh cue-invariance cell: Meta English
implicit cues measured 0.3566 fitting SD against the 0.25 maximum. This supports
a narrower claim that purpose is linearly decodable in these controlled sets;
it does not support cue-invariant purpose attribution.

There is no pending authorized activation extraction. A new extraction should
follow only after a prospective redesign is specified and frozen. Do not score
`naturalistic_transfer.jsonl` during development.

## Shared files

- `analyze.py` provides general development diagnostics.
- `fit.py` is the historical fixed-C, last-token diagnostic.
- `cue_invariant.py`, `cue_balanced.py`, and `cue_constrained.py` implement the
  reusable estimators used by later workflows.
- `format_matched_controls.jsonl`, `cue_invariant_confirmatory_controls.jsonl`,
  and `cue_constrained_confirmatory_controls.jsonl` are stable generated banks.
- `metrics/` contains the authoritative compact outputs referenced by reports.

## Run convention

Commands in workflow handoffs are run from `contrastive/`. Scripts resolve
their inputs relative to the repository, so moving into a workflow directory is
not required. GPU notebooks use `MyDrive/lang-ea/` as durable Colab transport;
their handoffs list the exact uploads required.

The package integrity boundary still includes `pyproject.toml` and `uv.lock`, so
probe-only dependencies should remain ephemeral rather than being added to the
project lock.
