# Contrastive v2 generation

This directory is a self-contained project for building the bilingual
contrastive probe set:

```text
localized topics + paired containers
                ↓
blinded frame calibration → passed evidence
                ↓
template-blind content specifications
                ↓
audited, immutable generation jobs      ← preflight (no model calls)
                ↓
verified provider handoff
                ↓
generate → cross-verify → render → publish   ← generate (calls models)
                ↓
verify → group by frame family → freeze controls   ← postprocess (offline)
                ↓
held-out model comprehension → passed evidence
```

`preflight` makes no provider requests: it plans, renders, and audits exact
model-visible requests without calling a model. The `generate` runner consumes a
verified pair of preflight runs, calls the providers, cross-verifies on the
non-authoring model, and renders and publishes accepted items. Corpus-wide
post-generation structural gates are implemented by `postprocess`. A development
forecast from already-collected stand-in activations is recorded in
`probe_check/development/layer_selection/RESULT_explicit_contract_8b.md`; production-corpus probe gates and
naturalistic transfer remain pending.

The third untouched confirmation is complete and rejects
`balanced_penalty_256`. Every fresh purpose-ordering gate passed on both 8B
stand-ins, but Meta's English implicit-cue effect was 0.3566 fitting SD against
the frozen 0.25 maximum. The bank is spent, no cue-invariant production probe is
adopted, and naturalistic transfer remains sealed. See
`probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md`.
The combined confirmation archive and prospective generation redesign are in
`docs/PROBE_REDESIGN_ARCHIVE_2026_08_06.md`.

## Status

Catalog version is **2.3.2**, and no corpus has been generated under it.

The pipeline has run end-to-end under 2.2.0. The reference run is `demo16`,
2026-08-01: a 320-job development slice (16 topics × 10 content specs × 2
languages) authored jointly by Qwen and Gemini, each cross-verifying the other.

| Result | Count |
|---|---|
| Accepted | 306 / 320 |
| Exhausted | 14 / 320 |
| Published | 306 content + 1224 rendered items |
| Provider errors | 0 |

Exhaustion still skews Japanese, though far less than in the earlier 60-job
slice: of the jobs whose language is recorded in the manifest, 9 JA against
2 EN. Because `postprocess --production` requires **every** planned job to be
accepted, a 4.4% exhaustion rate blocks a production corpus on its own; raise
`generation.max_regenerations` before generating the corpus that will carry the
calibration evidence, since that block is hashed into
`generation_settings_sha256`.

**Held-out comprehension was run against demo16 on 2026-08-02 and failed.**
Readers were Claude Sonnet 5 and GPT-5.4 through OpenRouter (see
`comprehension_check/`). Every cell cleared the 0.60 implicit floor, the
above-chance bound, and explicit-beats-implicit; all four failed on implicit
class symmetry, because deploy-side frames were read correctly much less often
than eval-side frames. Errors were concentrated in five of the ten implicit
pairs and 12 of GPT-5.4's 14 deploy errors reproduced in Claude, which pointed
at the stimulus rather than the readers. Claude was additionally found unfit as
an instrument — 0.67 on **explicit** deploy frames, where the lexeme is present,
against GPT-5.4's 0.96.

Catalog 2.3.0 repairs those five pairs (p06–p10), changing only noun phrases and
shared non-cue elements so the implicit cue-verb inventory, the frozen lexical
controls, and the span caps are untouched. Two of the five were cases where an
element required to be neutral was not: `第三者` in p10 carries a
third-party-verification sense in Japanese and is p01's own eval cue, and p09's
shared `参照` leans evaluative. The report is archived at
`calibration/heldout_comprehension_demo16_20260802_failed.json`.

The same revision carries two QC repairs found in demo16's exhaustion records.
The verification prompts' wrapper-exclusion paragraph did not name option
markers, so a verifier rejected a valid item over renderer-added `(ア)` labels
and a container prior-context line; the paragraph now states positively that
only the fields named in the authoring requirements are authored. And the four
free-response specs — `content_p05`, `p06`, `p07`, `p10` — now require CONTENT to
supply every fact the answer depends on and to not state the answer itself,
with `content_p07` gaining the "from CONTENT alone" clause the other three
already had in both languages. "Not answerable from content alone" was the
largest systematic rejection class in demo16 and fell entirely inside those four
specs.

The new wording is **unreviewed**. Semantic approval is `pending` with a waiver,
frame-only recognizability is waived, and held-out comprehension is `pending`
again but is currently covered by the scoped `heldout_comprehension` waiver.
Catalog 2.3.2 therefore passes production preflight with notices; a fresh
development corpus is still required to retire the waiver.

## Active inputs

- `topics.yaml` and `topics.ja.yaml`: paired English and Japanese topic views.
- `containers.yaml` and `containers/`: renderer-owned eval/deploy structures.
- `purpose_frames.yaml`: five frozen bilingual frame families and 20 unique
  matched purpose contrasts, each declaring the `cue_locus` that carries the
  contrast in both languages.
- `validation/lexical_controls.yaml`: frozen mention-only and crossed-cue
  legacy diagnostics for both the explicit evaluation/deployment lexemes and
  the implicit scrutiny-versus-use cues.
- `validation/format_matched_controls.yaml`: the first full-prompt 2x2
  purpose-by-cue bank. It has been scored and is now development-only: the
  unprojected explicit-only probe preserved purpose ordering but failed the
  lexical-cue-effect bound.
- `validation/cue_invariant_confirmatory_controls.yaml`: the second,
  text-disjoint full-prompt 2x2 bank. It is scored and development-only after
  rejecting SVD90 on explicit-lexeme cue effects.
- `validation/cue_balanced_development_augmentation.yaml`: provenance-bound
  development expansion of both spent banks across three additional cue-line
  wordings/placements. It is not fresh confirmation evidence; its Japanese
  template wording is pending native review.
- `validation/naturalistic_transfer.yaml`: frozen out-of-distribution transfer
  contexts.
- `inputs/topic_grid.csv`: hash-pinned source provenance for the topic grid.
- `prompts/content_generation_*.txt`: model-visible authoring instructions.
- `prompts/content_scaffolding.yaml`: localized slot and response guidance.
- `prompts/verification_*_{en,ja}.txt`: blind extraction and comparison prompts.
- `configs/pre_generation.yaml`: model-neutral run configuration.
- `configs/generation_run.yaml`: runner configuration (run pair, concurrency,
  verification settings, all-in budget, lease timeout).
- `calibration/*.json`: current frame-only and held-out-model evidence status.
- `configs/providers/`: `qwen_openrouter.yaml`, `gemini_vertex.yaml`, and
  `provider.example.yaml` as a template.

The generator receives a localized topic and one shared content specification.
It never receives a container, class label, purpose frame, body template,
nuisance profile, design note, or machine identifier. One accepted payload
feeds both members of two distinct eval/deploy wrapper pairs (four rendered
items total).

## Active commands

Run these from `contrastive/` — the `contrastive-data` entry point is declared in
this subproject, not at the repository root.

Create a development pre-generation run:

```powershell
uv run contrastive-data preflight `
  --provider-config configs/providers/<profile>.yaml `
  --run-id <run_id>
```

Development slices should sample topics evenly across the eight domains, since
each domain holds exactly five topics. Two per domain is 16 topics and 320 jobs:

```powershell
--topic t001 --topic t002 --topic t006 --topic t007 --topic t011 --topic t012 `
--topic t016 --topic t017 --topic t021 --topic t022 --topic t026 --topic t027 `
--topic t031 --topic t032 --topic t036 --topic t037
```

The original demo15 slice used 15 topics and left `workplace` with one topic
against two for every other domain, so per-domain counts were 40/40/40/40/40/40/40/20.

Add `--production` to require the complete bilingual grid (all 40 topics, all 10
content specs, both languages), seed support, usable credential references, and
current semantic approval plus passed current-catalog frame-only and held-out
model calibration evidence — except for any gate listed under `waived_gates` in
the run config (see "Resolved controls" below). `--production` rejects partial
slices; development mode is the only way to run a subset.

Create and score the blinded frame-only calibration packet:

```powershell
uv run contrastive-data calibration-packet `
  --output-dir data/calibration/<packet_id>

uv run contrastive-data calibration-check `
  data/calibration/<packet_id> ratings.jsonl `
  --raters confirmatory_raters.json `
  --output calibration/frame_calibration_passed.json
```

`confirmatory_raters.json` records a cohort ID, true attestations that the
cohort is new and independent from frame development, that the key and paired
counterparts were not shown, and one pseudonymous rater record with qualified
languages and a new-rater attestation for every rater ID in `ratings.jsonl`.

After a current-catalog development generation covering both languages, all
20 pairs, and both classes is postprocessed, classify
`heldout_comprehension_items.jsonl` with at least two mutually
lineage-disjoint calibration models, then gate the predictions.
`comprehension_check/` implements that classification against Claude Sonnet 5
and GPT-5.4 through OpenRouter and writes both inputs the gate needs; see
[`comprehension_check/README.md`](comprehension_check/README.md).

```powershell
uv run contrastive-data comprehension-check `
  data/processed/<dataset_id> predictions.jsonl `
  --models heldout_models.json `
  --output calibration/heldout_comprehension_passed.json
```

Point the pre-generation configuration at the passed reports before a
production preflight.

Generate, cross-verify, render, and publish a verified run pair:

```powershell
uv run contrastive-data generate `
  --run-dir-a runs/<run_a> --run-dir-b runs/<run_b>
```

Argument order is load-bearing: `--run-dir-a` becomes `generator_a`, and the
author/verifier split derives from that. Swapping the flags changes which model
authors every item. All bookkeeping (`state/`, `sessions/`, `lease.json`,
`stimulus_manifest.jsonl`) is written under run A; published items land under
each author's own `data/pilot/<run_id>/` tree, so a run pair produces output in
**both** trees.

Verify and assemble accepted outputs into an immutable, structurally complete
dataset. The current emitted contract blocks activation collection until the
prospective probe redesign is independently confirmed:

```powershell
uv run contrastive-data postprocess `
  --run-dir-a runs/<run_a> --run-dir-b runs/<run_b> `
  --output-dir data/processed/<dataset_id>
```

Add `--production` to require complete production-mode preflight runs and
acceptance of every planned job. The command makes no model calls. It writes
the fitting items, matched-pair index, five frame-family folds, EN/JA calibration
groups, untouched lexical and naturalistic validation sets, a probe-evaluation
contract, QC report, and hash manifest.

Verify a written run, or break a stale lease:

```powershell
uv run contrastive-data verify runs/<run_id>
uv run contrastive-data lease-release runs/<run_id>
```

Refresh or check the integrity lock:

```powershell
uv run python scripts/refresh_pre_generation_lock.py
```

Use `--write` only after reviewing intentional changes. `REQUEST_PROJECTION_FILES`
deliberately covers *every* locked implementation module, so any source edit
moves `request_projection_sha256` and forces approval back to `pending`,
regardless of whether the change could affect model-visible bytes.

### Credentials

- Qwen (OpenRouter): `OPENROUTER_API_KEY`, with a funded account.
- Gemini (Vertex): `gcloud auth application-default login`, then point
  `GOOGLE_APPLICATION_CREDENTIALS` at the resulting
  `application_default_credentials.json`. The file is a gate check only —
  requests are signed with `gcloud auth print-access-token`.

## Resolved controls and remaining prerequisites

**1. Japanese generated-content surface policy — resolved.**
The blanket Latin-letter ban was replaced with a declared English/Japanese
surface heuristic: at least 20 Japanese-script characters, an inclusive
ASCII-Latin share of at most 0.80 over the combined generated slots, and no
full-width digits. Embedded units, acronyms, option labels, and source code are
recorded diagnostics rather than automatic failures:

> 「ちょっとこの**PR**のコードレビューお願いできる？…対象の**PR**は#123で…」

Calibration over 78 parseable JA and 37 EN candidates found a JA minimum of 55
Japanese-script characters and maximum Latin share of 0.633333, while all EN
candidates had zero Japanese-script characters. Offline replay moves all 61
former JA surface failures through the new surface check; 13 still fail the
independent frozen-cue gate. This is intentionally scoped to the experiment's
two languages, not presented as general language identification.

The complete verification configuration is bound under the pair lease in
`runs/<run_id_a>/generation_policy.json`. Resumes with a different policy fail
before provider calls, and legacy runs with state but no policy artifact require
fresh run IDs.

Reproduce the archived calibration without provider calls:

```powershell
uv run python scripts/replay_japanese_surface.py --assert-slice-baseline
```

**2. The two human gates are waived; held-out comprehension is not.**
`configs/pre_generation.yaml` carries a `waived_gates` block waiving
`semantic_approval` (native-speaker semantic review) and `frame_only` (the
blinded human rater cohort). A waiver removes **only** the production block.
Every hash binding, schema check, and provenance comparison still runs, a
waived gate is reported as `pending` with an attached `waiver` rather than as
`passed`, and the record propagates into `preflight_report.json`
(`C15`, `C23`, and `C24_waived_gates`), the run notifications, `qc_report.json`,
and `dataset_manifest.json`. Delete an entry to re-arm its gate.

Because a waiver is not an approval, `prompts/pre_generation_semantic_approval.json`
stays at `status: pending` and the lock records the same. Runs carrying these
waivers have **unverified** EN–JA semantic equivalence and Japanese naturalness,
and **no** isolated-frame recognizability evidence; anything derived from them
inherits that status.

**2b. The cue lexicon now has two scopes and is bilingually symmetric.**
Lexicon `2.0.0` splits the broad request screen (`languages`) from a narrower
screen over model-authored text (`generated_content`), which must remain a
subset of it. Ordinary-domain polysemes — `認証` (authentication), `dataset`,
`compliance`, `research`, `study` — were rejecting valid generated items without
carrying any evaluation-versus-deployment signal, and are screened out of
requests only. Two bilingual asymmetries were closed at the same time: Japanese
now screens `運用`, the canonical deployment lexeme in the lexical controls
(previously only `配備` was listed, so generated Japanese could carry the
deployment cue unchecked), plus `コンプライアンス`, `プローブ` and `ルーブリック`;
and English matching now accepts regular inflection, so `evaluations`,
`assessments`, `tests` and `deployed` no longer slip past a stem-only match while
the Japanese substring screen catches every form. Replaying the demo15 run, 16 of
36 attempts that failed the old cue screen clear the new one; the other 20 carry
real contrast lexemes and still fail. See `PRE_GENERATION_PIPELINE.md` §Cue
policy.

**3. Held-out comprehension remains armed and still blocks production.** It is
the only remaining empirical check that the purpose frames land in complete
prompts, it is the sole diagnostic the probe contract's `selection_prohibitions`
leaves open, and it screens at 0.60 for a probe gate that must clear 0.65 per
language. It has been exercised once, against demo16 under catalog 2.2.0, and
failed on class symmetry; see "Status". `comprehension_check/` implements the
reader run, and a blinded packet is produced by `postprocess` for any
development corpus — 400 items, all 20 pairs, all 80 language × pair × class
cells at five each. Re-arming it against catalog 2.3.2 needs a fresh
development corpus, then classifications from two lineage-disjoint models.
Choose those models before scoring: the gate requires every declared model to
pass, so swapping one afterwards is selection on the gate outcome.

**4. Budget accounting is now two-layered.** Pre-generation authoring exposure
is still authorized per provider run. Generation startup additionally computes
author plus cross-verification calls for both purpose-masked pair-shared bodies
under baseline, regeneration,
maximum-output/schema/escalation, and transport-risk scenarios. The selected
all-in scenario must fit `configs/generation_run.yaml` before provider objects
are constructed. The same pre-call step conservatively checks the enlarged
verification requests and outputs against each verifier's context window.
Operators must still review the chosen basis and amount;
switching to a less conservative basis is an explicit authorization decision.

**5. Verifier transport failures no longer reject content.** A verified-QC
author candidate enters `verification_pending`. Cross-verification checks both
pair-shared bodies with the purpose frame masked, avoiding class-conditioned
acceptance, and gates topic/spec adherence, answerability, register,
distractor quality, and answer leakage. Transport exhaustion preserves the candidate, completed body
checks, and their hashes, then resumes at the same author attempt. Only a
completed verifier judgment can consume the candidate as a verification
rejection.

**6. The circuit breaker recovers.** It counts transport failures only, uses a
declared cooldown, admits one half-open recovery call, and clears on success.
The complete verification prompts, schemas, state schema, lease schema, and
runtime implementation are inside the integrity boundary.

## Provider configuration notes

Settings that were required to make the two subject models work, recorded so
they are not silently reverted:

- **`structured_output_mode: json_schema`** for Qwen. Under `prompt_only` the
  schema is never sent in any form, and Qwen returned labeled prose instead of
  JSON on 100% of attempts — failing both as author and as verifier.
- **`reasoning_effort: none`** for Qwen. Otherwise it spends the entire
  completion budget reasoning and returns `content: null` with
  `finish_reason: length`. Measured effect on an equivalent call: 4096
  completion tokens → 12.
- **`circuit_breaker_failures: 20`** and a 60-second recovery cooldown on both
  paid profiles.

## Documentation

- [`docs/PROBE_REDESIGN_ARCHIVE_2026_08_06.md`](docs/PROBE_REDESIGN_ARCHIVE_2026_08_06.md):
  current-status authority for all three confirmations, the claim boundary,
  status precedence, and the crossed term-aware redesign requirements.
- [`PRE_GENERATION_PIPELINE.md`](PRE_GENERATION_PIPELINE.md): architecture,
  invariants, artifact formats, gates, and handoff boundary.
- [`docs/POST_GENERATION_PIPELINE.md`](docs/POST_GENERATION_PIPELINE.md):
  immutable grouped assembly, frozen controls, and probe-evaluation contract.
- [`docs/SESSION_HANDOFF_2026_08_04.md`](docs/SESSION_HANDOFF_2026_08_04.md):
  wider operational state, open decisions, and implementation traps.
- [`probe_check/development/layer_selection/RESULT_explicit_contract_8b.md`](probe_check/development/layer_selection/RESULT_explicit_contract_8b.md):
  full two-model explicit-only contract result and reproducibility record.
- [`docs/GENERATION_RUNNER_PLAN.md`](docs/GENERATION_RUNNER_PLAN.md): design
  record for the implemented generation runner.
- [`docs/LOCALIZATION_REVIEW.md`](docs/LOCALIZATION_REVIEW.md): consolidated
  bilingual review evidence.
- [`docs/CUE_LOCUS_ENGLISH_ASYMMETRY.md`](docs/CUE_LOCUS_ENGLISH_ASYMMETRY.md):
  open design decision — English single-locus implicit frames underperform their
  Japanese counterparts, and the shared-locus rule makes it unfixable pair by
  pair.
- [`comprehension_check/README.md`](comprehension_check/README.md): the
  held-out comprehension reader runner, its blinding and counterbalancing
  choices, and the remaining operator steps.
- [`docs/HONEYPOT_PRE_GENERATION_REFERENCE.md`](docs/HONEYPOT_PRE_GENERATION_REFERENCE.md):
  the pre-generation control crosswalk used during design.

`runs/` and `data/` are gitignored except for their `.gitkeep` placeholders.
Generated items are local-only and are not preserved by a commit — archive them
deliberately if a run's output matters.
