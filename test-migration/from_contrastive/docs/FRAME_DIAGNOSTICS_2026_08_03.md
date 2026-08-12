# Frame diagnostics, 2026-08-03

**Headline: the frame question is now answered — `both`-locus is required for
English — but the catalog still should not go to full generation, because the
gate's own criteria now conflict.** Ten diagnostic experiments (7,594
classifications, $6.07 this session) found a large uncontrolled position effect
in the English gate, located the real per-pair weaknesses once it is removed,
showed that gate-screened frame repairs do not survive held-out content, and
then — using a screening panel disjoint from the gate readers — established that
the surviving distinction is cue *locus*, not wording. Sections 5b and 5c carry
the conclusions; sections 3–4 are the path that got there and overstate what
wording can achieve.

The most consequential finding is methodological: **screening frames against the
comprehension gate produces gains that largely vanish on unseen content.** Three
previous rounds of frame revision were run against this same gate. That is the
most likely explanation for the doc's standing puzzle that "two earlier rounds of
pair-level diagnosis did not replicate."

All experiments are diagnostic side runs — **not gate evidence**. They reuse
`demo16_v2` bodies and run `comprehension_check/classify.py` unmodified. No
catalog file, lock, or approval record was changed.

## 1. English has a large option-order position effect; Japanese has none

Pooling the first three experiments, English deploy items:

| option order | English deploy accuracy |
|---|---|
| `eval` listed first | **0.47** (84/178) |
| `deploy` listed first | **0.93** (198/212) |

English eval is 1.00 under both orders — the evaluation prior saturates it. So
the effect is confined to the deploy side, where the prior has to be overcome.

This is a **symptom of cue weakness, not a competing explanation**. Where the
frame is strong the effect disappears:

| frame | deploy, eval-first | deploy, deploy-first |
|---|---|---|
| English `verb_only` `apply` | 0.36 | 0.94 |
| English `both`-locus | 0.93 | 1.00 |
| Japanese (any arm tested) | 1.00 | 1.00 |

When the cue is insufficient the reader falls back on position plus the eval
prior; when it is sufficient, order stops mattering. Japanese never falls back.

**Consequence for the gate.** Option order is counterbalanced per item by hashing
the blind ID, which balances *in expectation* but not in any realized draw. At
5 items per pair per class, the realized order split is effectively random, and
it dominates the per-pair estimate. The observed English deploy rate of
0.66–0.78 is close to the midpoint of a bimodal 0.47/0.93 distribution. This is
sufficient on its own to explain non-replicating pair-level diagnosis.

**Every number below is order-balanced** — each item was run in both orders, by
salting the blind ID until it hashed to the wanted order.

## 2. Order-balanced per-pair diagnosis

English single-locus pairs and all Japanese pairs, own frames, own bodies,
n = 16 per cell.

English deploy (order-balanced):

| pair | locus | Grok | DeepSeek |
|---|---|---|---|
| p02 | `verb_only` | 0.81 | 0.81 |
| p04 | `verb_only` | 0.62 | 0.80 |
| p06 | `noun_only` | 0.81 | **0.13** |
| p07 | `noun_only` | **0.12** | 0.47 |
| p09 | `noun_only` | 0.62 | — |
| p10 | `verb_only` | 0.69 | — |

Two things stand out. **p07 at 0.12 for Grok is a severe regression** — the
recent English repair (`An operations team` → `A help desk agent`) made it far
worse than the 0.50 recorded before. And **the two readers disagree violently**:
p06 is 0.81 for Grok and 0.13 for DeepSeek; p07 is 0.12 for Grok and 0.47 for
DeepSeek. Both readers were calibrated to ≥0.95 on the explicit stratum.

Japanese (Grok, order-balanced) reproduces the real gate almost exactly —
simulated implicit gap **0.126** against the actual **0.12**, which validates the
simulation method:

| pair | deploy | | pair | deploy |
|---|---|---|---|---|
| p01 `both` | 0.88 | | p06 `noun_only` | 0.69 |
| p02 `verb_only` | 1.00 | | p07 `noun_only` | 1.00 |
| p03 `both` | 1.00 | | p08 `both` | 0.75 |
| p04 `verb_only` | 0.75 | | p09 `noun_only` | **0.56** |
| p05 `both` | 1.00 | | p10 `verb_only` | 0.62 |

The two Japanese pairs kept in the 2.3.0 decision (p09, p10) are the two weakest,
as that decision anticipated.

## 3. Frame repairs that look excellent

A hypothesis fit the English pattern: **single-locus pairs fail when the shared
(non-contrasting) element leans toward eval**, because it then reinforces the
prior on the deploy member. `p02`'s shared noun "The receiving team" is neutral
and p02 is the strongest single-locus pair; `p04`'s "An outside reader" and
`p10`'s "A third party" both lean eval and both are weak.

Repairs built on that hypothesis, screened on each pair's own bodies (Grok):

| pair | before | after | change |
|---|---|---|---|
| p04 | 0.62 | **1.00** | shared noun → "An external contact" |
| p07 | 0.12 | **1.00** | → "A compliance reviewer" / "The end customer will read" |
| p09 | 0.62 | **1.00** | → "A review team" / "The paying customer will receive" |
| p10 | 0.69 | **1.00** | shared noun → "The receiving party" |
| p02 | 0.81 | **1.00** | deploy verb → "act on" |

Every failing English pair went to ceiling. Japanese p10 reached 0.94 with a
span-cap-legal verb (用います → 使います).

## 4. …and mostly do not survive held-out content

The same chosen frames, applied to bodies from a **different** pair's content
spec:

| frame | Grok, own bodies | Grok, held-out | DeepSeek, held-out |
|---|---|---|---|
| p02 | 1.00 | **1.00** | **0.94** |
| p04 | 1.00 | 1.00 | 0.44 |
| p09 | 1.00 | 1.00 | 0.88 |
| p07 | 1.00 | 0.62 | 0.50 |
| p10 | 1.00 | 0.81 | 0.25 |
| JA p10 | 0.94 | 0.38 | 0.75 |

Three of six collapse on unseen content, and only **p02 survives both the
content change and the second reader**.

**Content type is not the explanation.** Holding one strong frame constant
("The receiving team will act on this response.") and varying the body across
all ten content specs gives deploy accuracy ≈ 1.00 everywhere, including the
three multiple-choice specs (1.00, 1.00, 0.81, mean 0.98). The bodies are not
the problem; the weaker frames are.

## 5. What this means

1. **The apparent repairs were substantially overfitting.** Screening a frame on
   the same ~15 bodies that measure it, against the same readers that gate it,
   inflates the estimate. The held-out drop is the size of that inflation.

2. **Three prior revision rounds were run the same way.** This is the most
   plausible mechanism behind the recorded non-replication, and it means the
   history of gate results overstates how much each round achieved.

3. **Per-pair frame quality is strongly reader-specific.** A 0.68 disagreement
   between two explicit-calibrated readers on the same frame and bodies means
   tuning against one reader carries almost no information about the other, and
   the gate requires both.

4. **One construction is validated**: a purpose-neutral shared noun plus "act
   on" (the p02 form), which held at 1.00 / 0.94 on held-out content across both
   readers. Everything else is unproven.

## 5b. Follow-up: the held-out failures were content-dependent, and `both` wins

Running the recommended protocol — a screening panel disjoint from the gate
readers (Kimi K2.5, Command A; both lineages disjoint from xAI and DeepSeek) —
resolved section 4 cleanly. The frames were re-tested on held-out bodies split by
content type.

**Free-response bodies** (`content_p05`, `content_p08`), order-balanced:

| arm | Grok | Kimi | Command A |
|---|---|---|---|
| `both` (new) | 1.00 | 1.00 | 1.00 |
| `both` (p03 catalog) | 1.00 | 1.00 | 1.00 |
| single-locus (p02 form) | 1.00 | 1.00 | 0.95 |

Everything passes. **Multiple-choice bodies** (`content_p01`, `content_p03`) —
the condition where every section-4 failure actually lived:

| arm | Grok deploy | gap | Kimi deploy | gap |
|---|---|---|---|---|
| `both` (new) | 0.96 | **0.04** | 1.00 | **0.00** |
| `both` (p03 catalog) | 1.00 | **0.00** | 0.96 | **0.04** |
| single-locus p02 | 0.88 | 0.12 fail | 0.88 | 0.12 fail |
| single-locus p10 | 0.88 | 0.12 fail | 0.83 | 0.17 fail |
| single-locus p07 | 0.88 | 0.12 fail | 0.91 | 0.09 |

Two independent readers agree closely — unlike the per-pair chaos in section 2,
which is what an uncontrolled instrument looks like versus a controlled one.

**Synthesis: multiple-choice bodies add eval pressure of roughly 0.12 on the
deploy side. `both`-locus frames have the margin to absorb it; single-locus
frames, even the best wording found across 8 verbs and two noun rounds, do not.**

This restores the original experiment-1 conclusion and kills option D properly:
redundancy is necessary, in the hard condition. Screening single-locus frames on
free-response bodies made them look fixed because that is the easy condition.

**Important mitigation**: only three content specs are multiple choice
(`content_p01`–`p03`), and of their pairs, p01 and p03 are already `both`. Just
**one** single-locus pair (p02) sits on multiple-choice content. So the
catalog-realistic projection is far better than the table above suggests.

## 5c. The real blocker: the gate penalises success

Projecting a fully repaired English catalog (single-locus frames on
free-response content at ~1.00, p02 at ~0.88, `both` pairs at ~0.98):

- implicit deploy ≈ **0.984**, eval 1.00 → class-symmetry gap ≈ 0.016 ✓
- implicit balanced accuracy ≈ **0.992**

But **English explicit balanced accuracy is exactly 1.000 for both readers**, and
the gate requires `explicit_greater_than_implicit` *strictly*
([calibration.py:691](../src/contrastive_generation/calibration.py#L691)). A
repaired English stratum passes that criterion by roughly one item out of 100,
and fails outright if a realized draw puts implicit at 1.00.

With eval pinned at 1.00 by the evaluation prior, the two armed criteria jointly
force English implicit deploy into a narrow band:

| criterion | implies |
|---|---|
| class symmetry ≤ 0.10 | deploy ≥ 0.90 |
| explicit (1.00) > implicit | deploy < 1.00 |

That is a 45–49 out of 50 target. **Fixing English well enough to pass the
symmetry gate pushes it into the failure mode of the other gate.** This is a
property of the criteria, not of the stimuli, and no amount of frame work
resolves it. Options: relax to `explicit >= implicit`, add a tolerance, make the
explicit stratum genuinely harder so it is not pinned at ceiling, or deliberately
calibrate implicit frames to mid-strength rather than maximum strength.

This should be settled before generation, because it determines what the frames
are being aimed at.

## 6. Recommended protocol change before any further frame work

The gate is currently being used as an optimization objective, which destroys
its status as an independent check. Three changes would fix that:

- **Screen and gate on disjoint reader sets.** Choose candidate frames against
  readers that are not the gate readers. This is the single highest-value change
  and costs nothing but reader selection.
- **Require held-out-content validation before adoption.** A frame is adopted
  only if its gain survives on bodies it was not screened on. This session shows
  that filter rejects three of six otherwise-perfect-looking repairs.
- **Order-balance any diagnostic.** Run each item in both option orders. Without
  it, English per-pair estimates are dominated by the position effect, at any
  n the gate can afford.

Given the reader disagreement, it is also worth asking whether a two-reader
unanimity requirement with a 0.10 class-symmetry cap is achievable at all for
English implicit frames, or whether the criterion itself needs revisiting. That
is a question about the gate, not about the frames, and it should be settled
before a fourth revision round.

## 7. Artifacts

Packets under `data/processed/{mechanism_p02,mechanism_p02_verbs,verbscreen,
locus4,perpair,repair,repair2,repair3,heldout,content}/`; predictions, per-item
raw responses, and keys under `comprehension_check/out/` (gitignored).

Readers: Grok 4.3 and DeepSeek V4 Pro (gate), plus Kimi K2.5 and Command A as a lineage-disjoint screening panel via comprehension_check/config.diagnostic.json. Gate readers are `@2026-08-03`, provider-pinned, no
structured-output downgrades, all `finish_reason: stop`. DeepSeek coverage is
incomplete in `locus4` (301/356) and `perpair` (~490/512) because of a sustained
upstream 429 on the Parasail shared pool; affected cells are marked by their n.

Session spend $6.07 across 5,095 new classifications. Ten packets: mechanism_p02, mechanism_p02_verbs, verbscreen, locus4, perpair, repair, repair2, repair3, heldout, content, robust, robustmc.
