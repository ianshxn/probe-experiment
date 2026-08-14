# English purpose x format probe MVP

This repository implements a development-only 2 x 2 experiment crossing:

- intended purpose: `evaluation` or `deployment`; and
- prompt format: `benchmark` or `casual`.

Each purpose-neutral canonical payload is rendered into all four cells. The
experiment compares a standard confounded probe with a probe trained after
format is decorrelated from purpose.

The project currently supports English only. It does not implement Japanese,
cue-polarity crossing, naturalistic transfer, or a confirmation-grade protocol.

## Status

Implemented:

- 16 English topics suitable for objectively answerable tasks;
- 10 development-only implicit purpose-frame pairs across five families;
- one benchmark MCQ renderer and one casual short-answer renderer;
- canonical-payload and rendered-item schemas;
- offline generation-request construction;
- OpenRouter generation with Grok 4.3;
- independent semantic validation with GPT-5.4;
- fail-closed acceptance, bounded regeneration, and complete attempt logs;
- deterministic complete-block rendering;
- structural validation by exact reconstruction;
- observational run manifests and hashes;
- fixed last-token activation extraction;
- grouped standard, decorrelated, and cross-format probe analyses; and
- unit and end-to-end smoke tests.

The unchanged migration snapshot and full design history are retained under
`test-migration/`. Live implementation code is at the repository root and does
not import from that snapshot.

## Experimental unit

One canonical payload produces:

| Intended purpose | Benchmark | Casual |
|---|---|---|
| Evaluation | evaluation + benchmark | evaluation + casual |
| Deployment | deployment + benchmark | deployment + casual |

The underlying context, question, correct answer, and distractor set are shared
across all four cells. The purpose sentence is identical across formats, and the
format template is identical across purposes.

Purpose-frame families are assigned deterministically by payload ID. All splits
hold out complete purpose families, so every payload block in a held-out family
is excluded from fitting.

## Installation

Core data-pipeline environment:

```powershell
uv sync
```

Development and probe analysis:

```powershell
uv sync --extra dev
```

Activation extraction additionally requires:

```powershell
uv sync --extra extract
```

## Local smoke workflow

Validate the included example payloads:

```powershell
uv run eval-format-mvp validate-payloads `
  --payloads data/example_payloads.jsonl
```

Render and validate the complete 2 x 2:

```powershell
uv run eval-format-mvp render `
  --payloads data/example_payloads.jsonl `
  --out outputs/rendered_items.jsonl `
  --manifest outputs/render_manifest.json

uv run eval-format-mvp validate-rendered `
  --payloads data/example_payloads.jsonl `
  --items outputs/rendered_items.jsonl `
  --report outputs/validation_report.json
```

The ten included payloads produce 40 rendered prompts, with ten items in each
purpose-by-format cell.

## Build the 80-payload MVP corpus

The intended development grid is 16 topics x 5 payloads = 80 canonical payload
blocks and 320 rendered prompts.

Write provider-neutral requests without making API calls:

```powershell
uv run eval-format-mvp requests `
  --per-topic 5 `
  --out outputs/generation_requests.jsonl
```

The checked-in profiles are ready for the recommended OpenRouter pairing:

| Role | Profile | Model |
|---|---|---|
| Generator | `configs/grok43_generator.openrouter.yaml` | `x-ai/grok-4.3` |
| Validator | `configs/gpt54_validator.openrouter.yaml` | `openai/gpt-5.4` |

Set the shared OpenRouter credential, then start with a small run:

```powershell
$env:OPENROUTER_API_KEY = "<your key>"

uv run eval-format-mvp generate `
  --generator-profile configs/grok43_generator.openrouter.yaml `
  --validator-profile configs/gpt54_validator.openrouter.yaml `
  --per-topic 5 `
  --max-generation-attempts 6 `
  --max-semantic-attempts 3 `
  --max-validator-attempts 2 `
  --limit 4 `
  --out data/payloads.grok43_gpt54.smoke.jsonl `
  --run-dir runs/grok43_gpt54_smoke
```

For the complete 80-payload MVP, use a new run directory and omit `--limit`:

```powershell
uv run eval-format-mvp generate `
  --generator-profile configs/grok43_generator.openrouter.yaml `
  --validator-profile configs/gpt54_validator.openrouter.yaml `
  --per-topic 5 `
  --max-generation-attempts 6 `
  --max-semantic-attempts 3 `
  --max-validator-attempts 2 `
  --out data/payloads.grok43_gpt54.full.jsonl `
  --run-dir runs/grok43_gpt54_full
```

Each Grok payload must pass deterministic checks and a high-confidence GPT
semantic judgment before it reaches the accepted payload JSONL. Generator-local
rejections and consistent semantic rejections have separate budgets. An internally
contradictory GPT judgment is retried against the same candidate and consumes
neither semantic attempt nor a new Grok generation. The validator never
rewrites the payload. When validating a regenerated candidate, GPT also receives
the cumulative concerns from earlier semantic rejections and must verify that the
new candidate explicitly resolves them. Grok uses `medium` reasoning with a
1,536-token output ceiling. GPT uses `medium` reasoning with a 4,096-token ceiling
for its structured judgment. In the same validator call, GPT searches for unstated
assumptions and reasonable counterexamples, solves the task without the options,
compares all three distractor pairs, and independently audits embedded artifacts
such as queries, formulas, code, charts, schedules, and rule sets against every
narrative claim. It also tests the claim under every reasonable interpretation or
execution semantic allowed by the context. An artifact contradiction or material
unstated semantic fails the relevant self-containment, correctness, or uniqueness
checks rather than being silently repaired, even when the proposed answer remains
correct. Candidates with option-dependent answers or contextually equivalent
alternatives are rejected.
If a validator response is truncated at its token ceiling, the same candidate is
retried within the existing validator-attempt budget. Distractor plausibility is
judged as a recognizable error pathway, separately from whether correct reasoning
definitively rules the distractor out. Symbolic construction tasks must define a
unique canonical form or ask for a uniquely determined property instead. Option
equivalence is judged
at the semantic granularity requested by the question, so identical outcomes do
not become distinct merely because they include different rationales. GPT records
the normalized direct answer of every distractor, and local validation rejects
duplicate normalized outcomes or distractors normalized to the correct answer.

The run is resumable by `payload_block_id`. It records:

```text
runs/<run-id>/run_config.json  # profiles and source hashes
runs/<run-id>/attempts.jsonl   # every accepted/rejected raw response
runs/<run-id>/run_report.json  # counts, exhaustion, and output hash
runs/<run-id>/errors/*.json    # one detailed record per caught attempt error
```

Error records are written for API/transport errors, structured-response parsing
errors, deterministic generator-output rejections, and terminal validator
inconsistency. Each includes the stage, exception, affected attempt, run and
attempt-log hashes, and whether the run stopped or continued. If OpenRouter
returned a response that could not be parsed, its raw response and provenance are
preserved in the error record before the exception is surfaced.

The run configuration is internally binding: changing a profile, prompt,
schema, topic catalog, or maximum-attempt rule requires a new run directory.
This prevents artifacts from incompatible configurations being mixed while
keeping the broader MVP hash policy development-friendly.

Both profiles require structured outputs, deny provider data collection, and
disable provider fallbacks. The generator and validator must come from different
model families. Limited runs are interleaved across topics, so `--limit 4` now
selects the first sample from four different topics rather than four samples from
the first topic. Remove `--limit` only after inspecting that smoke run.

The OpenRouter wire schema omits constraints unsupported by XGrammar (currently
`uniqueItems`); the complete checked-in JSON Schemas and deterministic validators
still enforce those constraints locally. API, routing, and schema-compilation
errors stop the run after transport retries and do not consume a generation or
semantic attempt. After correcting such an error, rerun the identical command to
resume.

Purpose neutrality is defined semantically as avoiding claims about experimental
use, evaluation, deployment, grading, or the intended audience. Topic-relevant
terms such as "code review" and "reviewer" are allowed. Deterministic payload
validation enforces structure, catalog membership, and answer uniqueness; GPT
alone judges semantic purpose and format leakage from context. Its complete
judgment and raw response remain in the attempt log for auditability.

## Activation extraction

The MVP preregisters the paper-motivated setting of Llama-3.1-8B-Instruct,
zero-based transformer block 24, and the last prompt token after applying the
chat template:

```powershell
uv run --extra extract eval-format-mvp extract `
  --items outputs/rendered_items.jsonl `
  --out-dir outputs/llama31_8b_layer24 `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --layer 24 `
  --batch-size 8
```

No model was downloaded or run as part of repository implementation. Extraction
requires suitable local GPU capacity and any model-access authorization required
by the model host.

The extractor writes:

```text
activations.npz
meta.jsonl
config.json
```

`config.json` records the stimulus hash, activation hash, model/revision,
zero-based layer semantics, position, item count, and hidden size.

## Probe analysis

```powershell
uv run --extra analysis eval-format-mvp analyze `
  --items outputs/rendered_items.jsonl `
  --activations outputs/llama31_8b_layer24/activations.npz `
  --out outputs/probe_results.json `
  --C 0.1
```

The fixed analysis reports:

| Regime | Training cells | Held-out evaluation |
|---|---|---|
| `standard_confounded` | Eval-Benchmark and Deploy-Casual | All four cells |
| `decorrelated` | All four cells | All four cells |
| `benchmark_to_casual` | Both purposes in benchmark format | Both purposes in casual format |
| `casual_to_benchmark` | Both purposes in casual format | Both purposes in benchmark format |

It also reports a format probe and within-payload eval-minus-deploy score deltas.
All results are out-of-fold under leave-one-purpose-family-out grouping. There is
no random row-level train/test split.

## Hash policy

Hashes are observational during MVP development. Manifests record the exact
payloads, catalogs, rendered text, model configuration, and activations used in
a run, but there is no repository-wide expected-hash lock.

Strict hash enforcement should be introduced only after the estimand, catalogs,
renderer, analysis, gates, and independent confirmation plan are frozen.

## Development-only boundaries

- The purpose frames were migrated from catalog 2.3.2 and have known lexical and
  noun-locus limitations.
- The included payloads are fixtures, not scientific evidence.
- A successful MVP result would demonstrate behavior of this controlled English
  dataset and one stand-in model only.
- It would not establish cue-invariant purpose attribution, naturalistic
  transfer, evaluation awareness, or a production monitoring method.
- The old confirmation banks remain spent, and the original naturalistic set
  remains sealed in the source project.

## Tests

```powershell
uv run --extra dev pytest
```

The tests cover catalog balance, the 80-request grid, deterministic four-cell
rendering, complete-block validation, tamper detection, and a synthetic example
where the standard probe follows a format shortcut while decorrelated training
recovers the independent purpose signal.
