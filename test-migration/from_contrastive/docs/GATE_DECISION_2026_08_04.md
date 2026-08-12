# Held-out comprehension: decision, and what it costs

**Status: decided and recorded. Production preflight passes.** Made 2026-08-04
under the study owner's standing authorisation, without a reviewer identity
supplied to the record.

## What was decided

`heldout_comprehension` is **waived**, per route 2 of
[`PATH_TO_GENERATION.md`](PATH_TO_GENERATION.md). The waiver is recorded in
`configs/pre_generation.yaml` alongside the two existing ones and is reversible:
deleting the block restores the hard stop.

Separately, the three `noun_only` implicit pairs were **rewritten** (catalog
2.3.2). Those two actions belong together and neither is sound alone — see
"Why both" below.

## Why the waiver

The decision rule was preregistered in `PATH_TO_GENERATION.md` before the
evidence existed:

> If a linear probe recovers the direction from English single-locus items
> despite the reader gap, then those items demonstrably carried the
> manipulation, the gate is a poor proxy for the probe, and **route 2 becomes
> well-supported**.

It fired. On two 8B models in the subject lineage, 1200 items,
leave-one-frame-family-out: English `verb_only` decodes at 0.907 (Meta) and
0.895 (Swallow), indistinguishable from `both`-locus, on items the gate scored
at 0.70 deploy-side. The gate's ordering is not reproduced at any of the four
depths sampled, on either model, in either language. Details in
[`../probe_check/development/baseline/RESULT_8b.md`](../probe_check/development/baseline/RESULT_8b.md).

The gate's own failure mode is independently explained: an explicit-stratum
screen of eight readers found every one at or near 1.00 on eval and below 1.00
on deploy, including on text that literally says "The deployment office will use
this response." The gate's deploy-side number measures the reader's evaluation
prior as much as the stimulus.

## What the waiver costs — precisely

`PATH_TO_GENERATION.md` scopes it, and the scope is narrower than it sounds:

> The gate exists so that a **null** probe result is interpretable, separating
> "the subject does not represent purpose" from "the stimulus never carried the
> manipulation." Waiving it does not weaken a positive result at all; it means a
> null must be reported as uninterpretable.

**This must go into the preregistration before any subject activations are
collected.** A null result on the affected stratum is uninterpretable under this
waiver. That is the whole cost, and it is not retroactively purchasable.

## Why both actions, not just the waiver

The same probe run fired the *other* branch of the same decision rule:

> If the probe also fails on single-locus items, the stimuli have a real problem
> that no structural fix in the current design can reach.

`noun_only` — p06, p07, p09 — sat at 0.500, 0.500 and 0.607 English balanced
accuracy, against 0.81–1.00 for every other pair, on both models and in both
languages. Waiving the gate on the strength of the `verb_only` result while
ignoring what the same run says about `noun_only` would be using the evidence
selectively. So the pairs were repaired rather than shipped.

The repair is scoped by the diagnosis, which is that the deficit is **not
grammatical**. Explicit `noun_only` frames decode at 0.82–0.88, so a
noun-carried contrast is recoverable in principle. What failed was this
construction: all three pairs shared a use-flavoured verb (*receive*, *handle*,
*process*; 受け取る, 処理する) that pinned both members to the deployment pole
regardless of the agent noun, while the two agent nouns were near-synonyms.
Each pair therefore got a neutral shared verb and genuinely polarised agent
nouns, with `cue_locus` and every machine constraint untouched. Full rationale
and the constraint checks are in `prompts/pre_generation_semantic_approval.json`.

## What is NOT established

- **The rewritten frames are unmeasured.** No instrument has scored them. The
  waiver ships frames whose only evidence is the diagnosis above. The cheapest
  checks are a re-gate (~$21) or a free offline probe check: the payloads are
  identical within a pair, so new-frame stimuli can be built by substituting the
  frame line into the existing `demo16_v2` items and re-running `probe_check/`
  with no generation at all.
- **The direction may be reading verbs, not purpose.** Every locus that decoded
  well carried a verb contrast; the only locus isolating the noun failed. That
  is the signature of a scrutiny-verb direction, which is exactly the
  alternative `CONTRASTIVE_SET_PURPOSE.md` names. The `implicit_cue` lexical
  controls were subsequently run; see the completed follow-ups below. Under
  implicit-only fitting they fail, while explicit-only passes the threshold-free
  diagnostic but not the formal joint machine-control gate. The waiver remains
  open pending a prospective recipe/control decision.
- **Naturalness and EN–JA equivalence of the new wording.** Unreviewed, like all
  of 2.3.x. The Japanese verb neutralisation (処理/受け取り → 見る/目を通す) needs a
  native check that it does not itself read as evaluative, which would defeat
  its purpose.
- **p02** English implicit deploy at 0.70 remains unrepaired and is now the
  largest untouched English single-locus weakness.

## State after this decision

`preflight --production` returns `planned` / `passed_with_notices`: 40 topics,
800 generation jobs, 3200 derived items, $53.24 estimated against a $300
authorisation. Three waivers now ride the run — `semantic_approval`,
`frame_only`, `heldout_comprehension` — and every one of them is a thing the
writeup has to declare.

Two mechanical notes carried forward from `PATH_TO_GENERATION.md` and still
unaddressed: `configs/generation_run.yaml` still points `run_dir_a/b` at the
existing `runs/demo16_v2_*`, so a real run needs fresh run IDs; and its
`budget.max_usd: 100` on `generation_retry_scenario` is a different basis from
the preflight's $300 / `transport_retry_risk_scenario`.

---

## Update, same day: the revisit condition has fired

The `implicit_cue` controls were run on the 1B stand-in that evening —
[`../probe_check/development/baseline/RESULT_controls_1b.md`](../probe_check/development/baseline/RESULT_controls_1b.md).
Threshold-free result: at layers 4, 6 and 8, in **both** languages, the fitted
direction scores *"deployment purpose + inspection verb"* above *"evaluation
purpose + use verb"* (ordering AUC 0.000-0.222 against a chance of 0.5). It
follows the scrutiny verb, not the purpose. `explicit_lexeme` passes the same
test, so the direction is not merely reading the evaluation/deployment lexemes.

This converges with the structural finding at 8B: every cue locus that decoded
well carried a verb contrast, and the one isolating the agent noun failed at
chance. Two independent routes, same conclusion.

**The waiver stood pending the preregistered follow-up.**
`probe_check/README.md` committed from the outset to treating a stand-in FAIL as
weak evidence. The same controls were subsequently run on both 8B models; the
completed status is recorded below.

Two consequences that do NOT depend on the stand-in and should be adopted now:

1. **Layer selection must pass the lexical controls, not merely maximise
   decodability.** The controls invert at exactly the shallow layers where
   decodability peaks. Freezing the layer on decodability alone actively selects
   the most verb-confounded layer. Adding a controls precondition to the layer
   freeze costs nothing today and cannot be added honestly once outcomes are
   seen.
2. **A fixed projection threshold does not transfer across item formats.**
   Control items sit -5.3 to +2.5 fitting-SDs from the fitting mean purely
   because they are bare prompts rather than frame-plus-payload. Stage-6 prompts
   are a third format, so scoring them with a threshold fitted on the
   contrastive corpus would partly compare formats rather than purposes. This
   needs per-format calibration or a within-item contrast.

Note finally that the catalog 2.3.2 rewrite deliberately **neutralised** the
shared verb in the three `noun_only` pairs. If the verb was carrying most of the
signal, that rewrite could make those pairs worse rather than better. The
new-frame extraction tests exactly that, offline and for free.

## Update: the rewrite itself was tested, and the stand-in cannot judge it

[`../probe_check/development/baseline/RESULT_newframes_1b.md`](../probe_check/development/baseline/RESULT_newframes_1b.md).
New-frame stimuli were built offline by substituting the frame line into the
existing `demo16_v2` payloads, so the comparison holds payload, folds, model and
code path exactly constant.

The comparison is **inconclusive, for a structural reason**: the 1B stand-in puts
the OLD `noun_only` frames at 0.625 English balanced accuracy, while both 8B
models put them at 0.534 and 0.568. The stand-in never exhibited the deficit the
rewrite targets, so it cannot show whether the rewrite fixes it.

What it does show is worth watching. The rewrite moved the two languages in
opposite directions: English effect size fell (0.40 -> 0.20 at layer 8, 0.52 ->
0.16 at layer 4) while Japanese rose (0.39 -> 0.54 at layer 8). That is the
direction the scrutiny-verb hypothesis predicts, since the rewrite deliberately
neutralised the shared verb, and English apparently leaned on it more.

If that asymmetry replicates at 8B it is a design problem, not a wording one: the
shared-cue_locus rule exists so neither language is more strongly cued than the
other, and this would reintroduce that confound through cue strength rather than
locus structure. The fix would then be stronger English agent nouns, NOT restoring
the use-flavoured verb -- restoring it would buy accuracy back by reinstating the
confound the controls just flagged, which is selection on the manipulation check.

## Update: the 8B GPU pass and explicit-only full forecast are complete

The requested 8B extraction and analyses are complete:

1. Under implicit-only fitting, `implicit_cue` inverts on both 8B models while
   `explicit_lexeme` passes; see
   [`../probe_check/development/baseline/RESULT_controls_8b.md`](../probe_check/development/baseline/RESULT_controls_8b.md).
2. The 2.3.2 rewrite improves Japanese but not English and widens the language
   gap; see
   [`../probe_check/development/baseline/RESULT_newframes_8b.md`](../probe_check/development/baseline/RESULT_newframes_8b.md).
3. The dense layer/position sweep rules out `frame` and identifies `last`/`mean`
   candidates; see
   [`../probe_check/development/layer_selection/RESULT_layerpos_8b.md`](../probe_check/development/layer_selection/RESULT_layerpos_8b.md).
4. The complete nested explicit-only forecast finds seven shared cells that pass
   every controlled representation gate plus threshold-free controls, but zero
   shared cells that pass the then-active bare-prompt absolute machine-control
   contract. The
   naturalistic set remains sealed; see
   [`../probe_check/development/layer_selection/RESULT_explicit_contract_8b.md`](../probe_check/development/layer_selection/RESULT_explicit_contract_8b.md).

The original automatic withdrawal condition is therefore recipe-dependent.
Implicit-only supports the narrower audience-scrutiny label; explicit-only is a
viable threshold-free candidate but is not contract-admissible as written. The
waiver remains recorded and unresolved. Before production subject-model
activation analysis, the study owner must prospectively declare the fitting
recipe and either retain the absolute control gate or validate an amended
paired/ordering gate on fresh controls.

## Historical update: the first prospective gate and bank were frozen

That decision was made before activation extraction. The confirmation used
explicit-only fitting at `mean` layer 6 on both 32-layer 8B development
models. A fresh 256-item full-format purpose × lexical-cue factorial was frozen
and unscored at that point. `last` layer 6 was diagnostic only.

The held-out-comprehension waiver remained recorded while that confirmation was
pending. Its completed failure and subsequent SVD90 sequence are recorded in
the updates below. Historical protocol:
[`../probe_check/confirmations/01_format_matched/CONFIRMATORY_CONTROL_EXTRACTION_HANDOFF.md`](../probe_check/confirmations/01_format_matched/CONFIRMATORY_CONTROL_EXTRACTION_HANDOFF.md).

## Historical update: first confirmation failed; second test was frozen

The first bank has now been scored. It passed every purpose-ordering cell on
both 8B models but failed all eight lexical-cue-effect bounds, so unprojected
explicit-only is rejected and the bank is spent. The waiver therefore remains
in force.

On that spent bank, leave-one-control-block-out development selected a fixed
SVD90 cue-nuisance projection. A second text-disjoint 256-item bank was frozen
and unscored at `mean` layer 6. A joint pass would have made that estimator eligible
for later production gates; a failure rejects it without rescue on the second
bank. The completed verdict is recorded in the next update.

## Update 2026-08-05: second confirmation rejects SVD90

The second bank has now been scored exactly once. All purpose-ordering and
implicit-cue effect gates passed on both models. Explicit-lexeme cue effects
failed in English and Japanese on both models (0.272â€“0.417 SD against the fixed
0.25 maximum), so the joint verdict rejects SVD90. Both banks are spent and
naturalistic transfer remains sealed.

The held-out-comprehension waiver therefore remains unresolved. There is no
adopted production probe recipe and no basis for unlocking Stage-6 scoring under
the stronger cue-invariant purpose claim. Full result:
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md).
