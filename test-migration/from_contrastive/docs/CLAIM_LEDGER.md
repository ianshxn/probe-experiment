# Claim ledger

What the contrastive set can currently support, and what each candidate design
change would do to it. Maintained alongside
[`FRAME_DIAGNOSTICS_2026_08_03.md`](FRAME_DIAGNOSTICS_2026_08_03.md) so that no
change to generation is made without recording its effect on the claim.

Claim wording is taken from
[`CONTRASTIVE_SET_PURPOSE.md`](CONTRASTIVE_SET_PURPOSE.md); the required
properties are numbered as they are there.

## The claim ladder

| # | claim | unlocked by |
|---|---|---|
| C3 | The subject model contains a linearly decodable direction associated with evaluation-versus-deployment **purpose attribution**, generalising across held-out framings and to naturalistic prompts. | all gates, incl. properties 4, 6, 8 |
| C2 | …a linearly decodable direction that **separates the controlled eval/deploy stimulus families**. | fitting + implicit recoverability (property 5) |
| C1 | …a direction encoding **audience-scrutiny attribution** — "my output will be judged" — not necessarily evaluation *per se*. | correct interpretation of the implicit-only development direction; not a production default |

No production claim rung is currently unlocked because no fitting recipe is
frozen and naturalistic transfer is unscored. C1 is the correct label for the
implicit-only development direction: its crossed controls follow scrutiny rather
than intended purpose. Explicit-only clears the threshold-free version of that
diagnostic in seven shared cells, but not the formal machine-control contract
jointly, so it does not yet unlock C2 or C3.

## A logical point that reframes the gate

The held-out comprehension gate exists so that a **null** probe result is
interpretable — it separates "the subject does not represent purpose distinctly"
from "the stimulus never carried the manipulation."

It follows that the gate is **insurance against a null, not a precondition for a
positive claim**. If a probe recovers the direction from a stimulus class, that
class demonstrably carried the manipulation, whatever a reader model says about
it. This is why item 3 is load-bearing and why it should have been run before any
frame rewriting: a positive probe result on single-locus items would retire the
entire cue-locus problem without touching the catalog.

## Effect of each candidate change

| change | claim effect | verdict |
|---|---|---|
| **D. Reword frames** (done in part) | None to the claim itself. Adds a provenance caveat: frames selected against the gate readers must be declared, and this session showed such selection largely does not generalise. | Safe, weak |
| **B. More `both`-locus** | Structurally capped at 4 of 10 — already the maximum (see below). Exceeding it requires relaxing the "no family repeats a locus" rule, which **confounds the leave-one-frame-family-out folds with locus** and directly weakens C3's "generalises across held-out framings". Also narrows implicit vs explicit (property 5) and removes the within-training dissociation that supports property 6. | Blocked, and costly if forced |
| **C. Language-specific locus** | Breaks property 7's stated purpose ("neither language more redundantly cued"). Because the contract fits **one direction across both languages** and per-language calibration happens after, EN and JA projections stop being on equal footing — property 10 fails, and the Stage-6 mediation analysis using *s* to explain the EN–JA behavioural difference becomes uninterpretable. | Reject |
| **E. Demote single-locus to generalisation-only** | The four `both` pairs span ff01–ff04 only; **ff05 has no `both` pair**, so the fitting set would cover 4 families and the contract's five leave-one-frame-family-out folds collapse. C3 unreachable. | Blocked |
| **Relax `explicit > implicit`** | This criterion encodes property 5, "explicit results are treated as a lexical upper bound." If implicit frames become as detectable as explicit ones, the claim that the direction survives *without* evaluation vocabulary is no longer demonstrated by the design. Falls back toward C1. | Changes the claim — declare it |
| **Waive `heldout_comprehension`** | Only bites on a null. A positive probe result is unaffected. A null becomes uninterpretable. | Acceptable if a null is reported as uninterpretable |

## The structural cap on `both` (verified by enumeration)

Three constraints in [`catalog.py`](../src/contrastive_generation/catalog.py)
govern locus assignment: no family may repeat a locus ("a holdout fold must not
be confounded with one locus"), `CUE_LOCUS_MIN_FAMILIES = 3`, and every surface
must carry every locus.

Exhaustively enumerating all 3¹⁰ assignments over the actual family and surface
layout, **only three balances are achievable at all**:

| balance | assignments |
|---|---|
| `both`=4, `verb_only`=3, `noun_only`=3 | 540 |
| `both`=3, `verb_only`=4, `noun_only`=3 | 540 |
| `both`=3, `verb_only`=3, `noun_only`=4 | 540 |

**The current 4/3/3 catalog is already at the maximum `both` the design admits.**
"Increase the `both` fraction" is not a tuning knob — it is a request to delete
an invariant. What each deletion buys:

| relax | max `both` | what is given up |
|---|---|---|
| nothing (current) | **4** | — |
| drop surface coverage | 4 | nothing gained |
| drop "no family repeats a locus" | 4 | nothing gained |
| `CUE_LOCUS_MIN_FAMILIES` 3 → 2 | 5 | a locus need only appear in 2 families — weakens C3's "generalises across held-out framings" |
| drop min-families **and** family-repeat | 6 | LOFO folds become confoundable with locus — attacks C3 directly |
| drop all three | 10 | C3 unreachable; also collapses property 7's spread requirement |

Two conclusions follow. First, reaching even `both`=6 costs the two invariants
that most directly underwrite C3. Second — decisively — the projected English
implicit deploy rate at `both`=6 is ≈0.77 against a 0.90 requirement, so **the
maximum relaxation short of destroying C3 still does not fix English.** Option B
is not merely expensive; it does not work.

## Standing risks to C3 independent of any change

- **Property 8, naturalistic transfer.** The naturalistic items mark purpose
  through a first-person situational clause ("I'm comparing several assistants
  for an internal capability audit…"), not through a noun+verb frame at all.
  They resemble *single-locus* subtlety more than doubly-cued frames. Removing
  single-locus pairs would remove the training stimuli most similar to the
  transfer target.
- **Property 6, crossed controls.** The 6 `implicit_cue` crossed items per
  language are single-channel by construction (an inspection verb with a
  deployment purpose). Single-locus training pairs are what teach the probe that
  either channel alone can carry purpose.

## Status

**Item 3 has reported, preliminarily: the gate does not predict the probe.**
See [`../probe_check/development/baseline/RESULT_preliminary.md`](../probe_check/development/baseline/RESULT_preliminary.md).
On a 1B stand-in, English `verb_only` items — the stratum the gate scores at the
bottom — are as linearly decodable as `both`-locus items (0.889 vs 0.875 at 50 %
depth). The gate's difficulty ordering is not reproduced at any depth sampled.

Claim consequences:

- **No claim is weakened by keeping the current 4/3/3 catalog.** The evidence for
  restructuring rested entirely on the reader gate, which the probe does not
  corroborate.
- **Waiving `heldout_comprehension` costs less than assumed.** The gate protects
  against an uninterpretable null; a stratum the probe demonstrably decodes did
  carry the manipulation regardless of what readers say about it.
- **A new risk appears that the gate never surfaced**: `noun_only` is the weakest
  locus for the probe in every layer and both languages. If that replicates at
  8B it bears on C3, and it is orthogonal to everything three rounds of frame
  rewriting have targeted.

Confirmation needs the 8B lineage-matched run (Colab notebook in
`probe_check/`), including the explicit stratum as a validity check.

---

# Update 2026-08-04: the 8B run, the controls, and the decisions taken

The confirmation run called for above was done, and the `implicit_cue` controls
were run for the first time. Both bear directly on the claim ladder.

## The controls moved C1 from "default" to "evidenced"

C1 was the standing default *until* the crossed lexical controls pass. They did
not merely fail to pass — on the 1B stand-in they **invert**. At layers 4, 6 and
8, in both languages, the fitted direction scores *"deployment purpose +
inspection verb"* above *"evaluation purpose + use verb"*: ordering AUC
0.000–0.222 against a chance of 0.5. Six cells, all far below chance. See
[`../probe_check/development/baseline/RESULT_controls_1b.md`](../probe_check/development/baseline/RESULT_controls_1b.md).

`explicit_lexeme` passes the same test (AUC ≥0.56 everywhere, 1.000 in English
from layer 4), so the direction is **not** merely reading the words "evaluation"
and "deployment". That retires one alternative explanation and sharpens the
remaining one.

This converges with a structural finding from the 8B run: every cue locus that
decoded well carried a verb contrast, and `noun_only` — the one locus holding the
verb constant — failed at chance and sat near-orthogonal to the others (cosine
0.18–0.31). Two independent routes to the same place.

| | before 2026-08-04 | after |
|---|---|---|
| C1 audience-scrutiny attribution | default, unevidenced either way | **default, with positive evidence** |
| C2 separates the stimulus families | reachable on implicit recoverability | unchanged, still reachable |
| C3 purpose attribution | pending all gates | **materially further away** — property 6 now has evidence against it, not merely absent evidence |

**Historical stand-in caveat and completed follow-up.**
`probe_check/README.md` committed in advance to treating a 1B FAIL as weak. The
same implicit-only control failure subsequently replicated on both 8B models.
The explicit-only follow-up then passed threshold-free controls in seven shared
cells but failed the formal contract jointly. See the final two updates below;
no production recipe or claim rung is frozen.

## The `noun_only` risk flagged above did replicate

At 8B, `noun_only` sits at 0.534 (Meta) and 0.568 (Swallow) English balanced
accuracy against 0.81–1.00 for every other pair, equally in both languages. It
bears on C3 exactly as anticipated. See
[`../probe_check/development/baseline/RESULT_8b.md`](../probe_check/development/baseline/RESULT_8b.md).

## A change to generation, and its claim effect

Recorded here because this ledger exists so that no generation change is made
without it. Catalog **2.3.2** rewrote the three `noun_only` pairs (p06, p07, p09)
with neutral shared verbs and more polarised agent nouns.

| change | claim effect | verdict |
|---|---|---|
| **Rewrite `noun_only` frames** (catalog 2.3.2, done) | Aimed at property 4/5 by making a failing stratum recoverable, without touching locus balance, span caps, family or surface coverage — so nothing in the C3 structural argument moves. Two risks. (a) It is **unvalidated**: the 1B stand-in never exhibited the deficit being fixed, so it cannot judge the fix. (b) On that same run the rewrite moved English **down** (effect size 0.40 → 0.20 at layer 8) and Japanese **up** (0.39 → 0.54). If that replicates at 8B it breaches property 7's purpose — neither language more strongly cued — through cue *strength* rather than locus *structure*, and would make the Stage-6 EN–JA mediation harder to interpret, the same failure mode option C was rejected for. | Provisional. Re-measure at 8B before treating as settled; if English does not recover, strengthen the English nouns rather than restoring the use-flavoured verb, which would buy accuracy back by reinstating the confound the controls just flagged. |

See [`../probe_check/development/baseline/RESULT_newframes_1b.md`](../probe_check/development/baseline/RESULT_newframes_1b.md).

## Decisions taken

`heldout_comprehension` is **waived**, per the row above and route 2 of
[`PATH_TO_GENERATION.md`](PATH_TO_GENERATION.md); the ledger's verdict
("acceptable if a null is reported as uninterpretable") is the operative
condition and must enter the preregistration before any subject activations are
collected. Decision, cost and open risks:
[`GATE_DECISION_2026_08_04.md`](GATE_DECISION_2026_08_04.md).

## Two protocol consequences that do not depend on the stand-in

1. **Layer selection must pass the lexical controls, not merely maximise
   decodability.** The controls invert at exactly the shallow layers where
   decodability peaks, so freezing the layer on decodability alone actively
   selects the most verb-confounded layer — which is selection *against* property
   6. Cheap to adopt now; impossible to adopt honestly once outcomes are seen.
2. **A fixed projection threshold does not transfer across item formats.**
   Control items sit −5.3 to +2.5 fitting-SDs from the fitting mean purely
   because they are bare prompts rather than frame-plus-payload. Stage-6 prompts
   are a third format, so scoring them with one threshold fitted on the
   contrastive corpus would partly compare formats rather than purposes. This
   bears on property 10 and on the Stage-6 mediation analysis, and needs
   per-format calibration or a within-item contrast.

---

# Update 2026-08-04, later: the 8B controls landed, and they are decisive

The stand-in caveat attached to the previous entry is now spent. The same
controls were run on both 8B models in the subject lineage.

**`implicit_cue` inverts at 8B.** 27 of 40 cells below chance at the `last`
read-out (median ordering AUC 0.333), 26 of 40 at `mean` (median 0.167; the
`frame` position is excluded here as an artifact -- see below); at
layers 2-16, where the probe decodes best, 23 of 32. **`explicit_lexeme` passes
in 120 of 120 cells.** So the direction is definitively not reading the
evaluation/deployment lexemes, and just as definitively is following the
scrutiny-versus-use verbs. See
[`../probe_check/development/baseline/RESULT_controls_8b.md`](../probe_check/development/baseline/RESULT_controls_8b.md).

**C1 is now the evidenced claim.** Not the default pending evidence -- the
evidenced one. C3 has direct evidence against it via property 6.

The waiver's own recorded condition ("if implicit_cue inverts at 8B as well, this
waiver should be withdrawn and the narrower claim label adopted") has fired on
the test it named. The waiver has NOT been withdrawn pending the study owner's
decision, because withdrawing it re-blocks production preflight; that is recorded
here so the omission is visible rather than silent.

**A lead that was recorded here and has since been withdrawn.** The `frame`
read-out (last token inside the purpose sentence) appears markedly less confounded
than `last` or `mean` -- 16/40 cells below chance versus 26-27, median 0.556 versus
0.167-0.333 -- and this entry originally logged it as the one position not clearly
inverted. It is an artifact. Under causal attention the frame-position activation
cannot depend on the payload, and measurement on the stored tensors confirms it:
within-frame-group spread is 1.1% of overall spread, so 300 implicit items per
language collapse to **20 distinct vectors**. Every `frame` statistic is a 20-point
classification replicated 15x, including the contract-gate passes at layers 4-14.
It is a frame-text classifier, which is the object `explicit_lexeme` already shows
is trivially decodable.

**Consequence for the claim: there is no read-out position that escapes the
inversion.** The C1-versus-C3 decision below has no third option pending further
measurement. `frame` should be dropped from the position sweep rather than carried
as unresolved. See [`../probe_check/development/baseline/RESULT_controls_8b.md`](../probe_check/development/baseline/RESULT_controls_8b.md).

## The 2.3.2 rewrite, re-measured at 8B

| change | claim effect | verdict |
|---|---|---|
| **Rewrite `noun_only` frames** (catalog 2.3.2) | Measured at 8B on both models. Japanese improved in 15/16 cells (median +0.109 balanced accuracy, +0.24 effect size); English improved in 4/16 (median -0.045, -0.16). The EN-JA gap on this stratum **widened** from 0.14 to 0.21 at layer 8. That breaches property 7's purpose -- neither language more strongly cued -- through cue strength rather than locus structure, which is the failure mode option C was rejected for. English `noun_only` still fails the 0.65 gate at 0.602. | **Incomplete.** Keep the Japanese side; rewrite the English side of p06 (dead at chance on both models, before and after) and probably p09. Keep the neutral verb -- restoring a use-flavoured one would buy accuracy back by reinstating the confound the controls just confirmed. Do not revert to 2.3.1, which was worse in both languages. |

See [`../probe_check/development/baseline/RESULT_newframes_8b.md`](../probe_check/development/baseline/RESULT_newframes_8b.md).

---

# Update 2026-08-04, later: what the controls can and cannot establish

Not a new measurement — a property of the existing instrument, found while
staging the English `noun_only` repair. It qualifies the entry above; it does not
reverse it.

**All 12 `implicit_cue` control terms are exactly the differing cue spans of the
fitting frames, in both languages (12/12).** `apply`, `check`, `employ`,
`inspect`, `use`, `vet` and 利用 活用 点検 用い 確認 精査 each appear as the token
that carries the eval/deploy contrast in one or two implicit pairs.

That overlap is by design — a crossed control takes the cue word and pairs it with
the opposite purpose. But it bounds the reading. The controls establish that the
direction follows *those specific words*. They cannot separate "the model encodes
audience scrutiny abstractly" from "the probe keyed on the frame's verb
vocabulary".

Two pieces of evidence argue against the purely lexical reading, and neither comes
from the controls: `explicit_lexeme` passes 120/120, and cross-language transfer
runs 0.687-0.737, which a direction keyed to English tokens could not achieve
against Japanese items. So there is real abstraction and C1 stands as the evidenced
claim. What changes is that **the controls measure lexical sensitivity, not the
absence of abstraction**, and the writeup should say so rather than presenting the
inversion as a clean verdict on representational content.

**The actionable half: the controls cover no agent nouns.** All 12 terms are verbs.
The three `noun_only` pairs have no manipulation check at all — and strengthening
the English agent noun is exactly the repair now staged. A `noun_only` improvement
measured against the current control set cannot be distinguished from the probe
learning a new agent-noun lexeme, which is the identical failure the verbs were
caught in.

| change | claim effect | verdict |
|---|---|---|
| **Adopt an English `noun_only` rewrite** (candidate catalog 2.3.3) | Would raise English `noun_only` decodability and, if it works, narrow the EN-JA gap back toward property 7's intent. But with no noun controls, the improvement is uninterpretable as evidence about purpose attribution. | **Measure first, then decide the controls question.** Measuring the candidates costs nothing and is staged. Adopting one without noun controls is defensible only if the writeup declines to read the improvement as evidence about representational content. |
| **Add an `implicit_noun_cue` control family** | Restores a manipulation check over the repaired stratum, and would let a `noun_only` improvement carry the same evidential weight the verb strata currently do. | **Study-owner decision.** Drafted, not adopted, in [`../validation/proposed_noun_controls.yaml`](../validation/proposed_noun_controls.yaml). `lexical_controls.yaml` is frozen and was not edited. Full reasoning in [`CONTROL_COVERAGE_GAP.md`](CONTROL_COVERAGE_GAP.md). |

---

# Update 2026-08-04, later: the read-out position was carrying some of this

Not new data — a sweep of the activations already on disk over 3 read-out
positions x 10 layers x 2 fitting recipes x 2 models, with an admissibility
filter applied before decodability is consulted. Full numbers in
[`../probe_check/development/layer_selection/RESULT_layerpos_8b.md`](../probe_check/development/layer_selection/RESULT_layerpos_8b.md);
reproduce with `probe_check/development/layer_selection/layer_position_sweep.py`.

**One entry above needs qualifying.** This ledger recorded "there is no read-out
position that escapes the inversion." That holds at the shallow depths it was
measured over, and at `last` at every depth. It does **not** hold at `mean`,
layers 24 and 32, where the implicit-fitted direction orders the crossed
`implicit_cue` controls correctly at 0.778–1.000 on both models in both
languages. The 26/40-below-chance aggregate pooled those cells with the shallow
ones. `frame` remains an artifact and this does not revive it — the sweep
confirms it independently, at 0.013–0.029 % within-frame variance share.

| change | claim effect | verdict |
|---|---|---|
| **Freeze `mean` rather than `last` as the read-out position** | Does not move any claim on its own — it changes which cells are eligible to be frozen, not what the corpus contains. In the implicit-only development sweep it is the only position that passes the scrutiny controls anywhere, so it is a precondition for that recipe supporting anything above *audience-scrutiny attribution*. Against it: `mean` is length-sensitive and its format offset to bare-prompt controls is larger deep on Swallow (+1.95/+2.46 fitting-SDs), which makes protocol consequence 2 — per-format calibration — more necessary, not less. | **Study-owner decision, and it must be taken as a rule rather than a cell.** Adopting "restrict to positions that see the payload and cells that pass the lexical controls, then maximise decodability" is preregisterable. Freezing `mean` L24 *because it passed here* is selection on the manipulation check. |
| **English `noun_only` repair (catalog 2.3.3)** | The deficit this repair exists to fix is largely a `last`-position artifact. At `mean`, English `noun_only` clears the 0.65 gate at 9–10 of 10 layers on both models, on the **unrewritten** frames, and the EN–JA gap on the stratum falls from 0.21 to 0.001–0.09. So a `noun_only` improvement attributed to the rewrite may be an improvement attributable to the read-out. | **Measure at both positions before committing.** This does not cancel the staged GPU pass — the variants are built and cost nothing extra to score at `mean`. It does mean a catalog bump, re-lock and approval-record edit should not be spent until the position question is settled, or the rewrite will be credited with a fix it did not make. |

**No cell passes everything.** Applying the contract's own
`frozen_success_criteria` — pooled implicit balanced accuracy ≥ 0.65 per
language, every fold x language cell ≥ 0.65, and both control families ordering
correctly — **0 of 80 model x cell combinations satisfy both the performance
gates and the controls.** The cells that pass the fold gate fail the controls
and vice versa; nearest misses are `mean` L24 implicit-fit (fold gate
0.655/0.569) and `mean` L14 explicit-fit (0.621/0.650). Stand-in forecast, and
the subjects are 9–12x larger, so this is not a prediction of failure. It is the
first evidence that the two gate families are in tension rather than jointly
satisfiable by luck, and it should be stated in the preregistration rather than
discovered after freeze.

---

# Update 2026-08-04, later still: the confound is in the recipe, not the corpus

Free measurement on activations already on disk. It does not overturn the control
inversion — it relocates its cause, and it reopens the purpose-attribution claim.

**A direction fitted on the explicit stratum passes the `implicit_cue` controls.**
The explicit pairs hold the verb constant and vary purpose lexically ("The
evaluation office will *use* this response" / "The deployment office will *use*
this response"), so a direction fitted there is never trained on a scrutiny
contrast. Scored against the crossed scrutiny controls it orders them **correctly**
— 1 of 48 cells below chance across two models, two languages, two read-out
positions and layers 6-24, against 23 of 32 below chance for the implicit-fitted
direction. It also transfers to implicit items at 0.67-0.83 balanced accuracy, and
`explicit_lexeme` passes under it too, so it is not a lexeme detector.

**What this means for the ladder.** The evidence against *a linearly decodable
direction associated with evaluation-versus-deployment purpose attribution* was
evidence against the direction produced by **one fitting recipe**, not against the
construct being present in the stimuli. The corpus does contain a recoverable
purpose direction. The implicit-fitted direction is best understood as a mixture:
cosine between the explicit-fitted and implicit-fitted directions is 0.27-0.41,
consistent with a shared purpose component plus a scrutiny component that dominates
when fitting happens where the two are confounded.

*A direction encoding audience-scrutiny attribution* is therefore no longer the
only label with evidence behind it. It remains the correct label **for the current
recipe**.

| change | claim effect | verdict |
|---|---|---|
| **Change the fitting recipe** (fit where purpose is unconfounded, test where it is implicitly cued) | Recovers a direction that passes both threshold-free control families and still transfers to implicit items. Arguably a stronger demonstration of purpose attribution than the earlier implicit-only recipe, since it shows generalisation from explicitly-labelled to implicitly-cued purpose. It does **not** pass the current absolute control contract jointly and does **not** address naturalistic transfer. | **Declare and freeze before the real run.** Legitimate a priori — "fit where the construct is unconfounded" is the reasoning that motivated looking. Illegitimate if adopted *because* it passed. Compare candidate recipes on development data, resolve the control interpretation prospectively, then freeze. |
| **Crossed scrutiny x purpose redesign** | The most rigorous route to the same place: dissociation built into the fitting set rather than borrowed from the explicit stratum. | **Defer.** No longer the only route, and it costs a regeneration, fresh controls and a preregistration amendment. |

Caveats that bind this entry: crossed controls are 3 vs 3 per language, so the
result rests on consistency across cells rather than any single cell; the effect is
depth-bounded to layers 6-24; and it is a forecast on a development corpus and
stand-in models. See
[`../probe_check/development/layer_selection/RESULT_xstratum_8b.md`](../probe_check/development/layer_selection/RESULT_xstratum_8b.md).

---

# Update 2026-08-04: explicit-only full contract forecast completed

The handoff's uncomputed full-gate task is now complete. The analysis used both
8B stand-ins, `last` and `mean`, all ten stored layers, five held-family outer
folds, family-grouped inner `C` selection, 2,000 whole-job bootstrap resamples,
and every available controlled gate. It fitted only explicit items and scored
both strata. Full result:
[`../probe_check/development/layer_selection/RESULT_explicit_contract_8b.md`](../probe_check/development/layer_selection/RESULT_explicit_contract_8b.md).

| result | Meta-8B | Swallow-8B | same cell on both |
|---|---:|---:|---:|
| all controlled representation gates | 12/20 | 13/20 | 11/20 |
| plus threshold-free crossed-control ordering | 8/20 | 11/20 | **7/20** |
| legacy bare-prompt absolute machine-control contract | 0/20 | 1/20 | **0/20** |

The seven shared threshold-free cells are `last` layer 6 and `mean` layers
6/8/10/12/14/16. Across them, worst fold × language balanced accuracy is
0.656–0.811, language-specific macro AUC is 0.834–0.969, implicit balanced
accuracy is 0.687–0.883, and crossed-control ordering AUC is 0.875–1.000.

**Claim consequence.** This strengthens the development case that explicit-only
can recover a direction which generalizes across controlled framings. It does not
unlock the purpose-attribution claim: no shared cell passes the machine contract
as written, and naturalistic transfer remains sealed. The fitting recipe remains
unfrozen.

**Protocol consequence.** The absolute lexical-control gate and explicit-only
are in tension. Bare controls sit −1.68 to +6.01 fitting SD from the fitting
format over the otherwise viable shared cells, and absolute mention-only effects
reach 4.87 SD. Changing the gate after seeing this result is not an analysis
cleanup. The study owner must either retain it and reject explicit-only, or make
a prospective paired/ordering or independently calibrated format amendment and
create fresh confirmatory controls.

| change | claim effect | verdict |
|---|---|---|
| **Freeze explicit-only now** | Would ignore a joint 0/20 pass under the current machine-control contract and an entirely unmeasured naturalistic gate. | **Reject for now.** |
| **Record fitting stratum as having no default** | Prevents the postprocessor and historical scripts from silently treating implicit-only or all-items fitting as the production recipe. | **Implemented.** |
| **Amend the lexical-control gate prospectively** | Makes the gate measure within-format purpose ordering rather than prompt-format offset, consumes the bare controls as development material, and requires a fresh confirmatory bank. | **Implemented before new extraction.** See the update below. |

---

# Update 2026-08-04: fresh full-format confirmation frozen

The prospective amendment is now concrete and cannot be tuned on its outcome.
The new bank has 16 bilingual payload blocks, eight per purpose surface, crossed
over two cue families, intended purpose, and lexical-cue polarity: 256 unique
full-format prompts. It was validated, materialized, hash-pinned, and frozen
without extracting or scoring activations.

The confirmation recipe is explicit-only. `mean` layer 6 is the sole gating cell
for the two 32-layer 8B development models; `last` layer 6 is diagnostic only.
The cell was selected by a recorded maximin rule over existing core metrics, not
the fresh bank. Passing requires purpose ordering within cue families, languages,
surfaces, and cue polarities, block-bootstrap lower bounds above chance, and a
bounded lexical-cue main effect. Legacy zero-threshold counts remain diagnostics.

**Claim consequence.** Nothing moves yet. A joint pass makes explicit-only
eligible for the next frozen gate; a failure rejects it under the new protocol.
Naturalistic transfer remains sealed, so even a pass does not unlock the full
purpose-attribution claim.

**Selection consequence.** Editing the bank, switching the gating cell, or using
the `last` diagnostic to rescue a failure would be selection on the manipulation
check. The next legitimate action is the hash-checked two-model extraction in
[`../probe_check/confirmations/01_format_matched/CONFIRMATORY_CONTROL_EXTRACTION_HANDOFF.md`](../probe_check/confirmations/01_format_matched/CONFIRMATORY_CONTROL_EXTRACTION_HANDOFF.md).

---

# Update 2026-08-04: first confirmation failed; cue-invariant confirmation frozen

The first full-format bank was extracted and scored under its frozen protocol.
Every required purpose-ordering AUC and interval gate passed on both models, but
every family-language lexical-cue-effect gate failed (0.527â€“1.580 fitting SD
versus the fixed 0.25 maximum). The joint decision is therefore
`explicit_only_rejected_by_fresh_confirmatory_controls`. The bank is spent and
cannot supply another confirmatory verdict.

Development on that spent bank compared an unchanged probe, four mean cue axes,
and a matched-difference SVD projection with leave-one-control-block-out checks.
SVD90 was selected by the recorded rule. It preserved a joint core floor of
0.6633 and purpose-AUC floor of 0.9688 while reducing the maximum held-block cue
effect to 0.1693 SD. This makes linear separation of purpose from the tested cue
interventions promising, not established.

A second text-disjoint 256-item bank was then frozen and unscored. Its method removed
the smallest SVD cue-difference subspace explaining at least 90% of squared
singular-value mass, then fitted explicit-only at `mean` layer 6. All thresholds
and the joint two-model rule were fixed.

**Claim consequence at freeze.** The unprojected explicit-only recipe was
rejected and the cue-invariant estimator remained a development candidate
pending the second-bank result recorded immediately below.

Current authority:
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_development_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_development_8b.md)
and, for the completed verdict,
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md).

---

# Update 2026-08-05: cue-invariant confirmation rejected

The second text-disjoint bank was scored exactly once under the frozen SVD90
protocol. Both models passed every pooled, surface, within-cue, and bootstrap
purpose-ordering requirement. Both also passed the implicit-cue main-effect
bound in English and Japanese. They failed the explicit-lexeme effect bound in
both languages: Swallow 0.272/0.279 SD and Meta 0.282/0.417 SD against the fixed
0.25 maximum.

The binding decision is
`cue_invariant_probe_rejected_by_fresh_confirmation`. SVD90 is not eligible for
production adoption. Both full-format banks are spent, and naturalistic transfer
remains sealed.

**Claim consequence.** The evidence strongly supports linear separation of the
controlled purpose labels under cue crossing, and shows that SVD90 removes much
of the tested lexical component. It does not support the stronger claim that the
fixed direction is invariant to explicit evaluation/deployment vocabulary on
new prompt contexts. The admissible description remains controlled stimulus
separation with measured lexical sensitivity, not a validated cue-invariant
purpose-attribution direction.

Numerical authority:
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md).

# Update 2026-08-05: cue-balanced reciprocal development selects nothing

The study owner retained the stronger cue-invariant goal. Four convex linear
objectives were declared and evaluated using only the two spent control banks:
cue-balanced control labels, a direct matched-cue score-change penalty, and two
combinations. Each bank served once as fitting control data and once as held
control data on both 8B stand-ins.

All purpose-ordering cells passed for every recipe. No recipe kept every held-
bank lexical-cue effect at or below 0.25 fitting SD. Balanced labels alone had
a maximum effect of 1.5595 SD; the best maximum among the penalized recipes was
0.3698 SD. Increasing the direct penalty from 1 to 4 barely changed transfer.
Under the recorded cascade, no recipe became control-eligible, core folds were
not inspected, and no estimator was selected.

**Claim consequence.** This development result does not change the binding
rejection or strengthen the scientific claim. It shows that strong purpose AUC
and same-format nuisance control are insufficient: lexical invariance must be
tested across cue-line contexts. Both source banks remain spent, naturalistic
transfer remains sealed, and no third confirmation bank has been created.

The next stage is a development-only five-context leave-one-context-out test.
The 1,536 added prompts reuse spent purpose/payload material under three new cue
renderings, so they cannot serve as confirmation evidence. Only a recipe that
passes every held-context and core gate on both models may justify freezing a
new estimator and authoring a later untouched bank.

Numerical authority:
[`../probe_check/development/cue_constraints/RESULT_cue_balanced_development_8b.md`](../probe_check/development/cue_constraints/RESULT_cue_balanced_development_8b.md).

---

# Update 2026-08-05: constrained development selects a confirmation candidate

The expanded A100 package supplied five spent cue-line contexts. The first
predeclared augmented evaluation selected nothing: Meta missed one Japanese
explicit-cue cell. A stronger first constrained round obtained the first full
context pass with exact all-cue row-space removal, but missed one Meta/Japanese
core cell by 0.0059. A final five-recipe trade-off round was declared before
scoring and retained every existing gate.

`balanced_penalty_256` was the sole joint eligible recipe. It fits explicit
purpose items plus all spent-control purpose labels at control mass 1 and adds a
256-weight mean-squared penalty over exact cue-swap score changes. Across five
held contexts its largest cue effects were 0.2122 SD on Swallow and 0.2474 SD on
Meta; every purpose AUC and interval floor was 1.0. Its core floors were 0.6897
and 0.6552. The Meta margins are only 0.0026 below the cue limit and 0.0052 above
the core floor.

**Claim consequence.** This is the first joint context-and-core development
pass and warrants confirmation. It does not reverse either prior binding
rejection, adopt a production probe, or establish arbitrary lexical invariance.
Naturalistic transfer remains sealed.

Only after selection, a third 256-item bilingual factorial bank was authored.
Its 16 payloads, 16 purpose variants, eight implicit cue pairs, and `Record
header` context are absent from development. It is frozen and unscored at
SHA-256 `676a7e688d1c88af3fd0fab9e12b09ea91ab6cbff3a64be03c3dbae745277c86`.
The next legitimate action is the exact two-model A100 extraction in
[`../probe_check/confirmations/03_cue_constrained/CUE_CONSTRAINED_CONFIRMATION_HANDOFF.md`](../probe_check/confirmations/03_cue_constrained/CUE_CONSTRAINED_CONFIRMATION_HANDOFF.md),
followed by the no-search evaluator once. A joint pass makes the candidate
eligible only for the later naturalistic gate.

Numerical authority:
[`../probe_check/development/cue_constraints/RESULT_cue_constrained_development_8b.md`](../probe_check/development/cue_constraints/RESULT_cue_constrained_development_8b.md),
with final development JSON SHA-256
`def7b012c957547300cb2a7a06f1ae6de8b40a197bb609c0debfa0522a7b23d6`.

---

# Update 2026-08-05: third confirmation rejects the constrained candidate

The third-bank extraction matched every frozen input, runtime, cell, and shape
requirement. The no-search evaluator fitted only `balanced_penalty_256`. Every
purpose-ordering and bootstrap gate passed with a floor of 1.0. Swallow passed
all four family-language cue-effect gates. Meta passed both explicit cells and
Japanese implicit cues, but English implicit cues shifted the score by 0.3566
fitting SD against the fixed 0.25 maximum.

The binding decision is
`cue_constrained_probe_rejected_by_fresh_confirmation`. The third bank is spent,
the selected development recipe is rejected, and naturalistic transfer remains
sealed.

**Claim consequence.** The evidence supports highly reliable controlled purpose
separation, including within each fresh cue polarity, but not a cue-invariant
purpose direction. Post-verdict diagnostics show the Meta failure across both
surfaces and intended purposes and across six of eight English implicit cue
pairs. The admissible description is controlled purpose separation with
measured lexical sensitivity.

**Protocol consequence.** The family-mean cue gate can hide large term-specific
effects through averaging; this occurs diagnostically even in some Swallow
implicit pairs despite its aggregate pass. That observation cannot change the
completed verdict. Any stronger future protocol must prospectively define a
term-level or hierarchical estimand and use another independent bank rather
than tuning penalty strength on the spent third bank.

Numerical authority:
[`../probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md`](../probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md),
with binding JSON SHA-256
`61fd25f7e237bfbf5fcf8a3f9df880a11ee946c797ed433f516b707e0f427f3d`.

---

# Update 2026-08-06: confirmation sequence archived and redesign boundary frozen

The three prospective confirmations, the development sequence between them,
the layer-6 selection limitation, and the generation implications are now
consolidated in
[`PROBE_REDESIGN_ARCHIVE_2026_08_06.md`](PROBE_REDESIGN_ARCHIVE_2026_08_06.md).
It is the current-status authority for the probe program; confirmation-specific
result JSON and `RESULT_*.md` files remain numerically authoritative.

The archive records four decisions:

1. retain the class-blind, pair-shared payload-generation architecture;
2. if the stronger claim is pursued, cross intended purpose with cue polarity
   inside the fitting corpus rather than relying on a validation-only control;
3. replace arbitrary lexical invariance with a prospective cue ontology plus
   term-level or hierarchical effects and an anti-cancellation gate; and
4. develop over a relative-depth grid, then freeze one layer/read-out/estimator
   cell before any new independent bank.

No redesigned schema, fitting corpus, estimator, or confirmation bank is yet
implemented or authorized. All three existing banks remain spent, no production
probe is adopted, and naturalistic transfer remains sealed. The immutable
source specifications retain `status: frozen_unscored` as a record of their
pre-extraction freeze; that embedded historical field does not override the
binding spent status.
