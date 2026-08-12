# Contrastive set purpose

## Purpose

The contrastive set is an auxiliary measurement instrument for the larger
English–Japanese honeypot study. Its purpose is to provide controlled
calibration data for fitting and validating, separately for each subject model,
a single linear direction that measures how strongly a prompt is internally
represented as serving an **evaluation purpose rather than an ordinary
deployment purpose**.

The deployment condition is the matched reference pole for the evaluation
condition. The set is not intended to identify two independent evaluation and
deployment axes.

For an activation vector \(h\), the intended probe output is a scalar projection:

```text
s(h) = w_eval-deploy · h
```

Higher values are more evaluation-like and lower values are more
deployment-like. The direction and every analytical choice used to obtain it
must be frozen before examining the main-study behavioral outcomes.

## Role in the larger study

The main study tests whether prompt language changes resistance to a private
incentive to violate a stated rule. The contrastive set does not estimate that
behavioral language effect. Instead, it supplies an internal-representation
measure that can be attached to the frozen main-study prompts.

After the probe is frozen, each Stage-6 prompt receives a projection score
\(s\). The analysis can then test whether prompts represented as more
evaluation-like are associated with greater temptation resistance and whether
that relationship helps explain the English–Japanese behavioral difference.
Behavioral outcomes must never be used to select examples, fit the direction,
choose layers or token positions, tune probe hyperparameters, or set
thresholds.

## Direct construct

The construct directly manipulated by this dataset is:

> Evaluation-versus-deployment purpose attribution.

The dataset presents content-matched prompts whose evaluation and deployment
members differ only in the attributed purpose of the model's response.

The resulting direction may be described as a **candidate
evaluation-awareness-associated direction** only after it generalizes beyond
the controlled training contrast. The dataset alone does not establish
conscious awareness, strategic behavior, or a unique internal representation
of evaluation awareness.

The implicit frames attribute purpose through who receives the response and
what they do with it — inspect it, or use it. That contrast carries a
**scrutiny confound**: a model may represent "my output will be judged"
without representing "this is an evaluation." The `implicit_cue` lexical
controls are the instrument that separates the two, by crossing inspection
verbs with deployment purposes and use verbs with evaluation purposes. Until
those controls pass, the narrower label *audience-scrutiny attribution* is the
defensible one.

**Evidence as of 2026-08-04, and it points at the confound.** The `implicit_cue`
controls were run for the first time, on a 1B stand-in. They invert: at layers 4,
6 and 8, in both languages, the fitted direction scores "deployment purpose +
inspection verb" above "evaluation purpose + use verb" (ordering AUC 0.000-0.222
against a chance of 0.5). `explicit_lexeme` passes the same test, so the
direction is not merely reading the evaluation/deployment lexemes. A stand-in
FAIL is weak evidence by the asymmetry `probe_check/README.md` commits to, and
the decisive run is the same controls on the two 8B models. But
*audience-scrutiny attribution* is now the label with evidence behind it rather
than the label held by default. See
[`../probe_check/development/baseline/RESULT_controls_1b.md`](../probe_check/development/baseline/RESULT_controls_1b.md)
and [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md).

## Required properties

The contrastive set fulfills its purpose only if:

1. Purpose attribution is the sole within-pair textual difference.
2. Generated task content is identical within each matched pair.
3. Answer format, speaker presence, prior context, length, scaffolding, topic,
   and other nuisance features cannot predict class independently.
4. Probe performance generalizes to held-out purpose-frame families, not only
   held-out topics or generated items.
5. The direction is recoverable from implicit as well as explicit purpose
   frames; explicit results are treated as a lexical upper bound.
6. Lexical controls show that the direction is not merely detecting evaluation
   and deployment vocabulary, nor the implicit scrutiny-versus-use vocabulary
   that carries the contrast when the explicit lexemes are absent.
7. The grammatical element carrying the contrast is declared, balanced across
   frame families and surfaces, and identical between the English and Japanese
   members of every pair, so neither language is more redundantly cued than the
   other.
8. The direction transfers to a frozen naturalistic set containing previously
   unseen, coherent evaluation and deployment contexts.
9. English and Japanese frames pass independent recognizability, semantic
   equivalence, and pragmatic-coherence review.
10. English and Japanese projection scores are calibrated for meaningful
    comparison within each subject model.
11. Statistical evaluation respects the shared payload, wrapper pair, frame
    family, topic, content specification, and language as grouped rather than
    independent observations.

## Intended claim

If the required validation gates pass, the contrastive set can support:

> The subject model contains a linearly decodable activation direction
> associated with evaluation-versus-deployment purpose attribution, and that
> direction generalizes across held-out framings and to naturalistic prompts.

Before naturalistic and held-out-frame validation, the defensible claim is
narrower:

> The subject model contains a linearly decodable direction that separates the
> controlled evaluation-purpose and deployment-purpose stimulus families.

## Current readiness

The checked-in design implements the structural and operational requirements:
purpose-only matched pairs, balanced nuisances, shared generated content,
held-out frame families, lexical controls, naturalistic transfer, grouped
provenance, and frozen probe-success criteria. Generation verification is
class-blind: it checks both pair-shared bodies with the purpose frame masked and
requires topic/spec adherence, answerability, register quality, plausible
distractors, and no answer leakage.

The instrument is not yet empirically qualified. Catalog 2.3.2 lacks current
native review and passed calibration evidence, no current full corpus has been
generated, and no subject-model probe has been trained. Production preflight
currently proceeds only through recorded waivers; that is operational permission,
not empirical qualification.

The explicit-only 8B development forecast clears the controlled representation
gates plus threshold-free controls in seven shared cells. Its first prospective
full-format confirmation passed purpose ordering but failed the lexical-cue
main-effect bound on both models, rejecting the unprojected recipe. A
block-cross-fitted SVD90 nuisance projection was then tested once at `mean`
layer 6 on a second text-disjoint bank. Every purpose gate passed, but
explicit-lexeme cue effects exceeded 0.25 SD in both languages on both models.
SVD90 is rejected, both banks are spent, and no production probe recipe is
adopted. Naturalistic transfer remains sealed. See
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md).

The study owner retained the stronger goal. The first reciprocal comparison
selected nothing, after which a five-context development package tested
stronger soft and exact constraints. `balanced_penalty_256` became the first
recipe to pass every context and core gate on both stand-ins. The third
untouched confirmation subsequently rejected it: all purpose gates passed, but
Meta English implicit cues exceeded the lexical-effect limit. The bank is spent
and the stronger claim remains unavailable.

The consolidated interpretation, live status of all three banks, and
prospective crossed-fitting redesign are recorded in
[`PROBE_REDESIGN_ARCHIVE_2026_08_06.md`](PROBE_REDESIGN_ARCHIVE_2026_08_06.md).
That archive supersedes conversation-only recommendations; it does not claim
that the redesign has been implemented.

## Non-goals

The contrastive set is not intended to:

- identify separate evaluation and deployment axes from a binary contrast;
- demonstrate the main study's language-by-incentive behavioral effect;
- prove that a model is consciously aware that it is being evaluated;
- establish that a decoded direction causally controls behavior;
- replace the main-study comprehension, naturalness, or behavioral gates; or
- use main-study outcomes to optimize the probe.

Independent evaluation and deployment axes would require at least a third,
neutral or unknown-purpose condition, allowing evaluation-versus-neutral and
deployment-versus-neutral directions to be estimated separately.

## Relationship to project artifacts

- [`BLOCKER_1_FIX.md`](BLOCKER_1_FIX.md) specifies the matched purpose-only
  container design.
- [`PROBE_READINESS.md`](PROBE_READINESS.md) tracks threats to a defensible
  probing claim and must be kept current with the checked-in catalog.
- [`../PRE_GENERATION_PIPELINE.md`](../PRE_GENERATION_PIPELINE.md) specifies
  generation, provenance, validation, and publication mechanics.
- [`POST_GENERATION_PIPELINE.md`](POST_GENERATION_PIPELINE.md) specifies the
  grouped fitting export, frozen validation sets, and prespecified empirical
  success criteria.
- The larger study's Stage-6 plan specifies that each frozen prompt later
  receives a probe projection \(s\), with behavioral outcomes joined only after
  the probe and analysis are frozen.
