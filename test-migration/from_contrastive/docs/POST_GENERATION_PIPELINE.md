# Post-generation pipeline

## Purpose

`postprocess` converts a completed current-catalog generation pair into one
immutable, structurally complete dataset. It performs no model calls and does
not train a probe. The current evaluation contract blocks hidden-state
collection until a redesigned protocol is independently confirmed; a later
qualified protocol must still run separately for each subject model.

## Command

```powershell
uv run contrastive-data postprocess `
  --run-dir-a runs/<run_a> `
  --run-dir-b runs/<run_b> `
  --output-dir data/processed/<dataset_id>
```

Use `--production` for the scientific corpus. Production mode requires two
valid production preflight runs, their complete shared grid, non-mock distinct
providers, a completed generation result, and acceptance of every planned job.
Development mode accepts a nonempty subset for testing but labels the output as
nonproduction.

The output directory must not already exist. The command builds in a temporary
sibling directory and renames it only after every gate passes.

## Input verification

The postprocessor:

1. re-verifies both immutable preflight runs and their frozen source hashes;
2. requires identical non-provider grids;
3. verifies the bound generation-policy hash and run pair;
4. rejects a missing or non-completed generation result and revalidates its
   hashed stimulus manifest and complete per-job state bundle;
5. rehashes every accepted content and rendered artifact;
6. reconstructs every rendered text from the current container and accepted
   slot payload;
7. checks every payload field against the preflight derivation metadata;
8. requires all four planned rendered outputs for each accepted job;
9. requires the selected attempt to contain passing purpose-masked verification
   judgments for both pair-shared bodies;
10. requires complete eval/deploy twins sharing one content hash; and
11. rejects overlap between fitting text and either frozen validation set.

Before acceptance, generation cross-verifies the generated task through both
pair-shared bodies with the purpose frame masked. This prevents a verifier from
selecting content through evaluation-only or deployment-only framing; the
separate held-out comprehension gate evaluates the complete framed prompts.

This deliberately rejects the older pilot outputs produced under the
superseded wrapper catalog.

## Outputs

| Artifact | Role |
|---|---|
| `probe_items.jsonl` | Matched-generated fitting items with labels and complete grouping provenance |
| `matched_pairs.jsonl` | Eval/deploy twin index keyed by job and pair |
| `folds.json` | Five leave-one-frame-family-out splits |
| `language_calibration_groups.jsonl` | Corresponding EN/JA design cells |
| `lexical_controls.jsonl` | Frozen mention-only and crossed-cue diagnostics |
| `format_matched_controls.jsonl` | First 256-item full-prompt purpose × cue factorial; scored and development-only |
| `cue_invariant_confirmatory_controls.jsonl` | Second text-disjoint 256-item purpose × cue factorial; scored, spent, and development-only after rejecting SVD90 |
| `naturalistic_transfer.jsonl` | Frozen one-shot out-of-distribution validation prompts |
| `heldout_comprehension_items.jsonl` | Blinded deterministic sample of complete prompts for lineage-disjoint calibration models |
| `heldout_comprehension_key.jsonl` | Private labels and design strata for that sample |
| `heldout_comprehension_protocol.json` | Model-disjointness requirements and comprehension gates |
| `probe_evaluation_contract.json` | Fit protocol, prohibitions, claim boundary, and prespecified success criteria |
| `qc_report.json` | Structural results and explicit status of empirical gates |
| `dataset_manifest.json` | Source references and hashes of every emitted artifact |

The fitting rows include payload/job, pair, frame family, topic, content
specification, language, and artifact hashes so inference can respect shared
observations.

## Split and fitting rule

The outer validation unit is `frame_family_id`, never an individual item.
Layer, token position, pooling, regularization, preprocessing, and language
calibration must be selected by grouped inner validation using only the four
outer-training families.

The explicitness stratum used for fitting is a declared recipe choice with no
default. The allowed choices are `explicit_only`, `implicit_only`, and
`all_matched_items`. After the recipe is frozen, refit on exactly the stratum or
strata admitted by it; an explicit-only or implicit-only recipe must never be
silently expanded to all matched items. Fit one direction separately for each
subject model.

The 2026-08-04 development forecast qualified explicit-only for a fresh-control
confirmation, not for production adoption. The first prospective bank preserved
purpose ordering but failed the cue-effect bound on both models, rejecting the
unprojected recipe. On that now-spent bank, block-cross-fitted development
selected a fixed SVD90 cue-nuisance projection. Its prospective second-bank test
preserved every purpose gate but failed the explicit-lexeme cue-effect bound in
both languages on both models. SVD90 is rejected and no production probe recipe
is adopted. See
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md).

A subsequent reciprocal development comparison selected none. The completed
five-context follow-up then compared stronger soft and exact constraints and
selected `balanced_penalty_256`, the sole recipe to pass all context and core
gates on both 8B models. The third untouched confirmation then rejected it on
Meta English implicit cues despite perfect purpose-gate passage. All three
banks are spent and naturalistic transfer remains sealed. See
[`../probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md`](../probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md).

Lexical controls and naturalistic transfer items are not fitting or tuning
data. Main-study behavioral outcomes are prohibited from every probe choice.

## Frozen empirical gates

The machine-readable contract requires:

- held-out balanced accuracy of at least 0.65 in every frame-family fold and
  language;
- macro ROC-AUC of at least 0.75 in each language, with the cluster-bootstrap
  95% lower bound above chance;
- implicit-frame balanced accuracy of at least 0.65 in each language, with its
  grouped lower confidence bound above chance;
- balanced accuracy of at least 0.65 for both external and institutional
  surfaces in each language, with each grouped lower confidence bound above
  chance; every pair is also reported as a diagnostic;
- on a future untouched full-format control bank, under a newly preregistered
  recipe if one is authorized: purpose AUC gates by cue family, language,
  surface, and cue polarity, grouped uncertainty by payload block, and
  prospectively defined term-level or hierarchical cue-effect gates with an
  anti-cancellation rule;
- legacy bare controls and all three full-format banks are spent development
  diagnostics and can never be reused as fresh confirmation evidence;
- at least 15/20 correct naturalistic items in each language, with at least
  7/10 correct in each class; and
- separate EN and JA passage of all primary gates. A pooled calibrated result
  cannot rescue a failing language.

The generated contract now blocks production-recipe adoption and activation
extraction. A future control threshold, relative-depth selection rule, and
gating cell must be frozen before any new confirmation activation is extracted
or inspected. The required redesign is recorded in
[`PROBE_REDESIGN_ARCHIVE_2026_08_06.md`](PROBE_REDESIGN_ARCHIVE_2026_08_06.md).

## Pre-probe calibration evidence

Two empirical manipulation checks are distinct from subject-model probing:

1. `calibration-packet` exports all 80 isolated bilingual purpose frames in a
   blinded random order. `calibration-check` requires at least five ratings per
   frame from a declared new confirmatory cohort, validates language
   qualifications and blinding/independence attestations, and enforces the
   original implicit lower/upper bounds, class symmetry, explicit minimum, and
   explicit-minus-implicit gap separately by language.
2. `postprocess` selects five complete prompts per language, pair, and class
   cell in production. Evidence must cover both languages, all 20 pairs, both
   classes, and at least one prompt per cell. At least two mutually
   lineage-disjoint models, also disjoint from both generators and subject
   models, classify these items. `comprehension-check` gates implicit
   recognizability, confidence above chance, class symmetry, and lower implicit
   than explicit performance separately for every model and language.

Both commands can write immutable reports. Frame-only evidence is catalog-bound.
Held-out-model evidence is additionally bound to the authoring semantic
projection, generation settings, and both generator model identities.
Production pre-generation refuses pending, failed, stale, or
generator-mismatched reports. Semantic approval is not treated as numerical
calibration evidence.

## Status boundary

A successful `postprocess` result currently means
`structurally_complete_probe_protocol_blocked`: dataset assembly and structural
gates succeeded, but activation collection is not authorized.
`qc_report.json` marks the empirical state
`blocked_pending_prospective_probe_redesign`. The emitted
`probe_evaluation_contract.json` records that no production recipe is adopted
and that a prospectively amended, independently confirmed control protocol is
required before extraction.
