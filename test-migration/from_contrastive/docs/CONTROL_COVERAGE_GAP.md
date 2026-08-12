# The lexical controls cover the verbs exactly and the nouns not at all

Written 4 Aug 2026, after the 8B control result. This does not overturn that
result — it bounds what the result can be read to mean, and it identifies a
blind spot that the planned `noun_only` repair would fall straight into.

## The measurement

`validation/lexical_controls.yaml` declares 12 `implicit_cue` cue terms, 6 per
language. Compared against the differing cue spans of the 10 implicit pairs in
`purpose_frames.yaml` (catalog 2.3.2), computed with the catalog's own
`cue_spans`:

| language | cue term | appears as the cue span of |
|---|---|---|
| en | `apply` | p01/deploy, p02/deploy |
| en | `check` | p04/eval, p05/eval |
| en | `employ` | p08/deploy, p10/deploy |
| en | `inspect` | p02/eval, p08/eval |
| en | `use` | p03/deploy |
| en | `vet` | p10/eval |
| ja | `利用` | p03/deploy, p04/deploy |
| ja | `活用` | p01/deploy, p02/deploy |
| ja | `点検` | p02/eval, p08/eval |
| ja | `用い` | p08/deploy, p10/deploy |
| ja | `確認` | p01/eval, p04/eval |
| ja | `精査` | p03/eval, p10/eval |

**12 of 12.** Every control cue term is exactly a token that carries the
eval/deploy contrast in the fitting set. Reproduce with the snippet in
[`../probe_check/README.md`](../probe_check/README.md).

## What this does and does not change

**It does not make the controls invalid.** A crossed control is *supposed* to
take the cue word and pair it with the opposite purpose — that is the whole
design. The overlap is intended, not accidental.

**It does bound the conclusion.** The controls establish that the direction
follows *those six verbs* against the intended purpose. They cannot separate:

- the model has an abstract audience-scrutiny representation that these verbs
  instantiate, from
- the probe has keyed on the specific lexical items the frame catalog happens to
  use.

Two pieces of evidence argue against the purely lexical reading and neither comes
from the controls: `explicit_lexeme` passes 120/120, so the direction is not
simply reading purpose vocabulary; and cross-language transfer runs 0.687–0.737
([`../probe_check/development/baseline/RESULT_8b.md`](../probe_check/development/baseline/RESULT_8b.md) §5), which a
direction keyed to English tokens could not achieve against Japanese items. So
there is real abstraction. The honest statement is that the controls demonstrate
lexical sensitivity and bound its extent, not that the direction is *only*
lexical.

This is a qualification on the strength of the C1 evidence, not a reversal of it.
C1 remains the evidenced claim and C3 still has evidence against it.

## The blind spot, which is the actionable part

**All 12 control terms are verbs. None is an agent noun.**

The three `noun_only` pairs — p06, p07, p09 — carry their contrast entirely in
the agent noun, and have no control coverage whatsoever. That matters right now,
because the repair everyone has converged on is *strengthening the English agent
noun* (see the variants in
[`../probe_check/development/baseline/frame_variants.json`](../probe_check/development/baseline/frame_variants.json)).

So the catalog is about to be repaired in the one place where the manipulation
check cannot see. If a strengthened noun raises English `noun_only` decodability,
the current control set cannot distinguish "the noun now carries audience
attribution" from "the probe learned a new agent-noun lexeme" — which is the
identical failure mode the verbs were just caught in.

## Recommendation

Extend the control set with an `implicit_noun_cue` family before adopting a
noun-strengthened catalog, mirroring the existing `implicit_cue` design: crossed
distractors pairing each eval agent noun with a deployment purpose and vice
versa, plus mention-only items.

`validation/lexical_controls.yaml` is inside the pre-generation integrity
boundary and its contents are frozen, so this session did **not** edit it. A
drafted, non-binding proposal sits in
[`../validation/proposed_noun_controls.yaml`](../validation/proposed_noun_controls.yaml)
— it is not referenced by any config, not loaded by any code path, and adopting
it is a study-owner decision with preregistration implications.

Note the sequencing consequence: this does not block generation, and it does not
block measuring the variants. It blocks *interpreting* a noun_only improvement as
evidence about purpose attribution.
