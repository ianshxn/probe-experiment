# Contrastive pre-generation pipeline

## Status and scope

This document describes the implemented version-2.1 pre-generation stage that
turns the current topic, purpose-frame, and container catalogs into immutable,
template-blind generation jobs. It plans, validates, serializes, and verifies
exact model requests without making network calls. Generation is implemented
by `generate`; offline probe-dataset assembly is implemented by `postprocess`.

Production planning is intentionally blocked while either catalog-bound
empirical manipulation check is pending: blinded human frame-only calibration
or complete-prompt comprehension by at least two lineage-disjoint held-out
models. Semantic approval is necessary but is not substituted for these
numerical checks.

The current catalog has 20 logical eval/deploy pairs, 40 logical containers,
and 80 localized container files (one English and one Japanese view per logical
container). `purpose_frames.yaml` is the authority for the five frozen frame
families and all bilingual frame texts.

## Scientific invariant

One accepted generation attempt produces the content shared by an eval/deploy
container pair. A generator is never asked to author one class independently
from the other.

```text
English and Japanese topic views
                +
forty logical matched-purpose containers
                │
                ▼
       compile ten content specs
                │
                ▼
 topic × content spec × language × sample
                │
                ▼
 preplan immutable, distinct-seed attempts
                │
                ▼
 exact monolingual requests, without a template
                │
                ▼
 one structured slot payload
       ┌──────────┴──────────┐
       ▼                     ▼
 wrapper pair A         wrapper pair B
   ┌───┴───┐             ┌───┴───┐
 eval   deploy         eval   deploy
```

With the complete current catalogs and one sample:

```text
40 topics × 10 content specs × 2 languages = 800 logical generation jobs
800 jobs × 3 preplanned attempts = 2,400 exact request artifacts
800 accepted payloads × 4 renderers = 3,200 derived items
```

The three-attempt figure follows the checked-in `max_regenerations: 2`. Attempts
are capacity, not three mandatory calls: the runner stops after the first
accepted payload. It never creates independent container-conditioned generation
jobs.

## Components

### Source catalogs

`catalog.py` remains the authority for source integrity. It validates:

- topic and container JSON Schemas;
- unique and stable identifiers;
- retained source-grid hashes;
- English/Japanese hash pins and invariant fields;
- localized registers, role structures, slot names, and response contracts;
- body-template/slot agreement and exactly one renderer-owned purpose slot;
- option and answer-key contracts;
- one eval and one deploy member in each pair;
- byte-identical bodies and nuisance profiles within each pair;
- the frozen 20-pair nuisance manifest and raw bilingual frame-length gates;
- exactly two wrapper pairs and four containers per content spec; and
- exclusion of numbered, XML, and first-person purpose framing.

### Content-spec compiler

`content_specs.py` groups containers by `content_spec_id` and compiles one
localized `ContentSpecRecord` per four-container content identity. It
deliberately does not use `containers.yaml.model_view`; purpose frames, matched
bodies, and nuisance profiles are renderer-private.

The compiler's model-payload allowlist is:

```text
register
role_structure
slots:
  - localized name
  - localized semantic description
response_contract
model_guidance
```

The following fields remain private:

```text
container_id
pair_id
content_spec_id
class
display_name
design_notes
purpose_frame
body_template
nuisance_profile
i18n
```

The compiler requires `p01`–`p20`, with exactly two pairs and four containers
for each of `content_p01`–`content_p10`.

### Request projector

`preflight.py` constructs requests from exactly:

```text
ModelTopic + ModelContentSpec + localized scaffold + generation parameters
```

`ModelTopic` and `ModelContentSpec` are narrow, immutable projections. They do
not expose machine topic labels, topic IDs, pair/spec IDs, container records,
classes, design notes, fixed templates, or item numbers. The request-rendering
function cannot accept a `TopicRecord`, `ContentSpecRecord`, or
`ContainerRecord`. This makes template blindness and private-field isolation
capability boundaries, rather than conventions enforced by remembering to
remove fields.

Each request artifact has four top-level fields:

```json
{
  "system": "localized authoring instruction",
  "user": "localized topic and content requirements",
  "response_schema": {},
  "generation": {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_output_tokens": 4096,
    "seed": 123
  }
}
```

Only the four-field `request` object is model-visible. Adapter routing is a
separate, frozen control plane; the provider adapter must never merge the
private manifest job or adapter metadata into the request.

### Planner

`pre_generation.py` expands the deterministic identity:

```text
(topic_id, content_spec_id, language, sample_index)
```

Container ID, class, selection-list position, and run ID do not participate in
logical generation identity. Each logical job contains
`max_regenerations + 1` preplanned attempts. Attempt zero preserves the logical
job's primary seed; later attempts have distinct deterministic seeds. The
primary seed also fixes renderer item numbering, so accepting a later attempt
cannot change wrapper assignment. Consequently:

- both classes use the same payload;
- reversing topic, content-spec, or language selection does not change retained
  job IDs or seeds;
- changing a wrapper changes derivation hashes but not request hashes;
- changing model, decoding settings, topic content, or content-spec content
  changes generation identity; and
- changing the run ID changes storage paths but not generation identity.

Increasing `max_regenerations` appends attempts without renaming the job,
changing attempt zero, or moving either derivation. No runner may improvise a
repair prompt or seed outside this frozen schedule.

### Integrity verifier and handoff

`verify_pre_generation_run()` rehashes every frozen input, validates every
schema, reconstructs every request, reruns request audits, recomputes seeds,
generation identities, renderer derivations, and bundle hashes, and rejects
missing or extra request files.

`prepare_generation_handoff()` performs that global verification before the
first provider call, then rechecks the plan, resolved configuration, and
manifest while capturing a sanitized, non-secret adapter profile. Its handoff
object rechecks those artifacts and rehashes one requested attempt immediately
before `submission_for(job_id, attempt_index)` returns an envelope. The adapter
uses the envelope's control-plane profile for routing and passes only
`envelope["request"]` to the model. API-key values are resolved from
`api_key_env` by the adapter. For Google profiles, `credential_env` is
only a non-secret readiness reference; the adapter authenticates through the
declared gcloud/ADC flow. Secret values are never persisted or returned by the
handoff.

Attempts must be released in order and each can be released only once by a
handoff object, including under concurrent callers. The handoff advances its
cursor when it releases an envelope; it does not observe response outcomes.
Provider-level transport retries reuse that already released envelope. The
runner requests the next seed only after the completed response is
malformed or quality-rejected, and must stop after success.

This provides accidental-change and corruption detection. It is tamper-evident,
not cryptographically authenticated: a malicious party able to rewrite every
artifact and recompute every hash is outside this threat model.

## Honeypot pre-generation crosswalk

The contrastive folder remains self-contained and imports no honeypot code, but
its pre-generation controls deliberately mirror the proven structure of that
pipeline:

| Honeypot pre-generation control | Contrastive analogue |
|---|---|
| Run-configuration schema | C01 |
| Family and blueprint schemas | C02 topic and C03 container schemas |
| Permanent identifier/catalog checks | C04 |
| English/Japanese invariant checks | C05, C06, and C09 |
| Registered family corpus validation | C07 registered corpus and C08 pair closure |
| Complete-grid and explicit development override | C10 |
| Selection consistency | C11 |
| Bilingual prompt-placeholder parity | C12 |
| Japanese scaffold contract | C13 exact Japanese semantic-surface audit |
| Scaffolding hash lock | C14 |
| Human bilingual semantic approval | C15, expanded to topics, compiled specs, and renderer templates |
| Rendered author-prompt cue screen | C16 plus the renderer cue contract |
| Job schema and uniqueness | C21, extended to preplanned attempts and every path class |
| Execution-time frozen-input rehash | C22 whole-run reconstruction and handoff-time checks |

Contrastive-specific additions are the private-value leakage screen (C17), a
typed template-blind capability boundary (C18), one shared payload for both
classes, context/cost/budget authorization (C20), and deterministic
regeneration requests frozen before the first call.

## Pre-generation gates

Every hard gate completes before an immutable generation manifest is published.
For a brand-new run ID, the writer must first create the run directory
exclusively; a concurrent writer cannot enter that namespace. The plan is
written last and serves as the run-completion marker. A process finding an
existing directory may reuse it only after the completed plan verifies and has
the same plan identity; an incomplete/in-progress directory fails closed.

| Gate | Implemented check | Failure behavior |
|---|---|---|
| C01 | Version-2 run-configuration schema | Hard stop |
| C02 | English and Japanese topic schemas and catalog integrity | Hard stop |
| C03 | Per-container schema and file/ID correspondence | Hard stop |
| C04 | Permanent topic, container, pair, and content-spec identities | Hard stop |
| C05 | English/Japanese topic invariants and source hash pins | Hard stop |
| C06 | English/Japanese container localization mappings and hash pins | Hard stop |
| C07 | Registered corpus: 40 topics, eight domains × five, 40 containers, ten specs | Hard stop |
| C08 | Pair closure: exactly one eval and one deploy member per pair | Hard stop |
| C09 | Purpose-only pair equality, frozen manifest, and two-pair/spec reuse | Hard stop |
| C10 | Production grid: all topics, all specs, both languages, exactly one sample | Hard stop in production; notice in development |
| C11 | Selection consistency and unknown-ID rejection | Hard stop |
| C12 | Exact EN/JA prompt-placeholder parity | Hard stop |
| C13 | Model-visible Japanese request surface: no NFKC Latin letters or full-width digits | Hard stop |
| C14 | Exact active-file set and per-file scaffold hash lock | Hard stop |
| C15 | Hash-bound researcher/native-speaker semantic approval | Hard stop in production; warning in development |
| C16 | Frozen cue lexicon (`languages` scope) over every exact model request; the narrower `generated_content` scope screens model-authored text in post-generation QC | Hard stop |
| C17 | Private field/value and fixed-template leakage audit | Hard stop |
| C18 | Template blindness by request-projector type boundary | Hard stop through tests and architecture |
| C19 | Provider structured-output, seed, context, output, revision, pricing, concurrency, credential-reference, and retry declarations | Invalid/unsafe contracts hard-stop; development records seed/replay and missing-variable notices; missing variables hard-stop in production and at handoff; `seed: none` hard-stops production |
| C20 | Per-attempt context bounds, run-level cost scenarios, and budget authorization | Hard stop |
| C21 | Exact configured Cartesian grid; job schema; contiguous attempt schedule; unique job, attempt, request, candidate, content, and derivation paths | Hard stop |
| C22 | Exclusive new-run ownership, immutable write, complete reconstruction, environment pin, and handoff-time plan/config/manifest/request checks | Hard stop |
| C23 | Current-catalog frame-only human calibration plus complete-prompt comprehension bound to current authoring semantics, generation settings, and generator identity | Pending/failed/stale/mismatched reports hard-stop production; warning in development |

There is also a renderer-specific cue gate:

- deployment wrappers must have zero frozen cue hits in both languages; and
- paired English/Japanese evaluation wrappers may differ by at most one cue hit.

Evaluation wrappers are allowed to contain evaluation framing because that fixed
framing is the intended construct. It never enters an authoring request.

## Japanese request policy

For every preplanned Japanese attempt, the audit NFKC-normalizes and examines:

- system text;
- user text;
- every user-defined structured-output property name;
- required-property names;
- string enums, constants, defaults, and examples;
- patterns and all schema annotations under pipeline control; and
- every recursively nested schema branch, including arrays, combinators,
  conditionals, definitions, and locally referenced subschemas.

Any ASCII or full-width Latin letter in that semantic surface is a hard failure.
Full-width digits are also rejected; numerals must be half-width.

The walker fails closed on unknown JSON-Schema extensions, external references,
unresolved local references, and unsupported anchor references. This prevents a
future schema-shape expansion from silently placing semantic text outside the
cue, language, or private-leakage audits.

JSON and JSON-Schema protocol vocabulary such as `type`, `object`, `string`,
`properties`, and the request-envelope keys are not natural-language
instructions and are excluded from the semantic-language scan. All
user-defined Japanese property names and values are localized.

Provider/model identifiers, endpoint URLs, authentication metadata, and JSON
wire-protocol field names belong to the adapter control plane. They may be
English, but they are not concatenated into `system`, `user`, or the semantic
structured-output schema and therefore are not model-facing prose.

An eventual provider adapter must not inject an English schema name, tool name,
or textual response instruction. If a provider translates JSON Schema into
additional prompt prose, that fully resolved provider payload needs the same
language and cue audits before submission.

### Generated Japanese content policy

Generated content has a different invariant from the pipeline-authored request.
Embedded ASCII is legitimate in Japanese SI units, acronyms, option labels, and
source code, so its presence is recorded rather than rejected categorically.
For this English/Japanese experiment, the generation runner NFKC-normalizes the
combined generated slot values and records:

- whole ASCII-Latin tokens;
- the number of characters matched by the frozen Japanese-script regex;
- `latin_share = latin_letters / (latin_letters + japanese_script_chars)`;
  and
- full-width digit hits from the original, pre-normalization surface.

The configured generated-content gate requires at least 20 Japanese-script
characters, an inclusive Latin share of at most 0.80, and no full-width digits.
A zero denominator records a share of 0.0 and still fails the Japanese-character
floor. The combined-candidate calculation is intentional: a source-code slot
may be entirely Latin while a separate slot supplies legitimate Japanese
framing.

This is an English-vs-Japanese surface heuristic, not a general-purpose language
identifier. The request/content asymmetry is also intentional: C13 controls
pipeline-authored instructions, while post-generation QC evaluates
model-authored subject matter.

## Cue policy

The cue lexicon lives in `prompts/generation_cue_lexicon.yaml`. Unlike the
active v2 authoring prompts do not:

- identify the task as a research dataset;
- mention a contrastive set or hidden class;
- show either container template; or
- enumerate prohibited evaluation words to the generator.

The frozen lexicon is enforced externally. This prevents the prohibition itself
from revealing the experimental setting.

### Two scopes

Lexicon version 2.0.0 carries two term lists, because the request screen and the
generated-content screen are doing different jobs:

- **`languages`** — the broad request screen (C16). Any hint of the study's
  vocabulary in a model-visible authoring request could steer the generator, so
  this list stays wide.
- **`generated_content`** — the narrower screen applied to model-authored task
  text. Generated content must exclude only terms that carry the
  evaluation-versus-deployment contrast or that name the experiment itself.

`generated_content` must remain a subset of `languages`; the lexicon validator
enforces this. Ordinary-domain words are deliberately absent from the narrow
scope. Screening `認証`, `dataset`, `compliance`, `research`, and `study` against
generated content rejected valid items over polysemy rather than cue leakage —
`認証` alone means *authentication* in any login or security topic and was the
most frequent Japanese cue hit observed.

### Bilingual symmetry

Because the estimand is an EN–JA contrast, the two screens must be equally
strict, or different content survives into each language and confounds the
comparison at the stimulus level. Two asymmetries were corrected:

- Japanese lacked entries for `compliance`, `probe`, and `rubric`, and — more
  seriously — screened `配備` while the lexical controls treat `運用` as the
  canonical Japanese deployment lexeme. Generated Japanese could therefore carry
  the deployment cue and pass. `運用` is now screened in both scopes.
- Japanese does not inflect, so a substring match on `評価` catches every form,
  while English stem matching let `evaluations`, `assessments`, `tests`, and
  `deployed` through. English matching now accepts regular inflection after a
  frozen stem. Forms that drop a stem vowel (`evaluating`, `probing`) remain
  enumerated in the lexicon.

English matching uses NFKC normalization, case folding, letter boundaries, and an
optional regular-inflection suffix, so `test` matches `tests` and `testing` but
still does not match `contest`, `testament`, or `greatest`. Japanese matching uses
NFKC-normalized occurrences.

## Scaffold lock and semantic approval

`prompts/pre_generation.lock.json` pins the exact active files used for request
construction and verification, including:

- both generation prompts;
- localized scaffolding;
- cue lexicon;
- approval record;
- run, provider, request, job, report, plan, lock, and approval schemas;
- catalog and content-spec compilers;
- request auditing and planning code;
- deterministic seed and hashing utilities; and
- renderer and CLI entrypoint code;
- `pyproject.toml`; and
- the exact `uv.lock`.

Topic and container files are not part of this global scaffold lock. Their
run-specific hashes are recorded in every plan, including every localized
container file.

The semantic approval record is separately bound to three hashes:

- the bilingual prompt/scaffolding bundle; and
- a canonical review projection containing paired topic display text, all
  compiled English/Japanese model content specs, and the fixed localized
  templates/item-number contracts that will consume their payloads; and
- every other non-circular locked implementation/runtime input, including all
  Python pipeline modules, schemas, cue-screen configuration, `pyproject.toml`,
  and the exact `uv.lock`. The approval record itself is excluded to avoid a
  circular hash, and the prompt/scaffolding prose is covered by the first hash.
  This conservative boundary includes hard-coded localized fragments,
  generated response-schema names, transitive parser behavior, and future
  locked helpers.

Its current status is `approved` under the study owner's explicit instruction
to assume human semantic approval for this implementation pass. The record
transparently states that no separate native-reviewer identity was supplied.
If the study later requires a named attestation, replace the assumption and
bind the replacement to the then-current three hashes.

Any later change to reviewed prompt prose, topic/spec text, renderer templates,
item-number contracts, or request-projection modules likewise invalidates an
approval and requires a new review.

After deliberately changing a locked implementation or scaffold:

```powershell
contrastive/.venv/Scripts/python.exe `
  contrastive/scripts/refresh_pre_generation_lock.py
```

The check should report that the lock is stale. Review the changes, ensure the
semantic approval is `pending` if any reviewed prompt, topic/spec text,
renderer template, item-number contract, or request-projection module changed,
then refresh:

```powershell
contrastive/.venv/Scripts/python.exe `
  contrastive/scripts/refresh_pre_generation_lock.py --write
```

The refresh helper refuses to silently carry an existing approval across a
change to any of the three bound semantic surfaces.

To approve a frozen bundle, a researcher and native Japanese reviewer edit
`pre_generation_semantic_approval.json` with:

- `status: "approved"`;
- both reviewer identities; and
- an ISO-8601 approval timestamp.

Then refresh the lock and rerun all tests. Any later change to reviewed prose,
catalog projections, or model-visible projection code invalidates that
approval.

## Provider-profile contract

Provider profiles remain separate from scientific catalogs. Version 2 requires:

- provider and exact model identifier;
- an asserted immutable model revision (obvious aliases such as `latest` are
  rejected; the adapter records the server-resolved model identifier);
- explicit generation route (`base_url` plus route for HTTP-compatible
  providers; literal project plus location for Google);
- native JSON-Schema structured-output mode and the pinned
  `draft_2020_12` dialect;
- declared seed behavior: `native`, `best_effort`, or `none`;
- context and output-token limits;
- explicit input/output prices, including an explicit zero if appropriate;
- `api_attempts`, the maximum total transport submissions for one frozen
  attempt including the initial submission, plus maximum concurrency; and
- credential environment-variable references rather than literal secrets.

Placeholder text is rejected recursively, not only at the top level. Keys such
as `api_key`, `token`, `secret`, and `password` cannot carry literal values.
URL userinfo, sensitive URL query parameters, bearer tokens, and secret-like
assignments in control-plane strings are also rejected. Credential values
themselves are never written into a plan.

Non-mock HTTP-compatible providers require HTTPS. API-key providers must name
the key environment variable; Google profiles must name a non-secret
`GOOGLE_APPLICATION_CREDENTIALS` reference. For Google, the referenced file
must exist. This is a local credential-source presence check, not proof that
Google will authenticate it; the adapter performs provider
authentication before spending calls. Development planning may record a
missing/unusable reference as a warning, but production planning and handoff
refuse it. HTTP routes must be safe relative paths or opaque labels; absolute
URLs, traversal segments, queries, fragments, control characters, invalid ports,
and known credential prefixes are rejected.

Routing identifiers are not credentials and may not be deferred through an
environment variable: they are frozen into `provider_execution_sha256`.
Provider-specific field sets are exclusive, so a Google profile cannot also
carry an HTTP endpoint and a mock profile cannot carry real authentication or
routing fields.

A missing credential variable is a warning in development and a hard failure in
production planning. It is always a hard failure at generation handoff.

`api_attempts` counts total transport submissions for one already frozen
sampling attempt, including the first submission. The provider profile
deliberately has no
`malformed_response_attempts`: once a model returns a completed but malformed
payload, that sampling attempt is consumed and the runner must advance to the
next preplanned attempt. This prevents hidden, unhashed model calls.

The adapter materializes the provider-specific wire payload and records it in
the call audit. The OpenAI-compatible JSON-Schema name is the language-neutral
identifier `0`; Japanese verification uses localized response-property names.
No tool name, fallback prompt, or prompt-only structured-output emulation is
allowed by the production profiles. Provider protocol keys remain distinct
from semantic prompt prose.

## Cost report

The plan records:

1. baseline generation calls, input-token estimates, expected output tokens,
   and cost;
2. the exact preplanned generation-attempt schedule;
3. a maximum-output scenario over every preplanned attempt; and
4. a conservative transport-retry exposure scenario.

The last figure is explicitly not called a guaranteed billing ceiling because
provider retry billing varies.

The plan refuses publication when the configured budget is below the selected
scenario. Non-mock profiles must use nonzero rates unless the configuration
explicitly records a zero-pricing waiver. Human-readable costs are rounded for
display, but authorization compares the exact decimal cost string, so
sub-micro-dollar overruns cannot round down into approval.

This pre-generation authorization covers author calls. Immediately before
generation, the runner performs a second all-in authorization over author plus
cross-verification calls. `configs/generation_run.yaml` selects the scenario
used for that hard gate. The bound `generation_policy.json` records the
breakdown, selected cost basis, budget, and a conservative verifier-context
authorization before a provider object is created.

Token estimates are reporting heuristics, not tokenizer-exact measurements:

- English uses approximately four characters per token;
- Japanese uses approximately 1.8 characters per token; and
- the semantic structured-output surface is included.

The hard context gate does not trust that heuristic. It uses UTF-8 request-byte
length as a conservative token upper bound, adds configured provider/chat
framing overhead and maximum output tokens, and compares the result with the
provider-declared context window.

## Run artifacts

`preflight` writes only under `runs/<run_id>/`:

```text
plan.json                 completion marker and top-level hashes
resolved_config.json      normalized selection and embedded provider profile
content_specs.json        private compiled EN/JA content-spec snapshot
manifest.jsonl            private generation jobs and renderer derivations
preflight_report.json     global and per-request gate evidence
notifications.json        development warnings and notices
approvals.json            approval snapshot bound to the run
requests/
  <job_id>_aNN.json       exact adapter-neutral attempt requests
```

Generated payloads and later derived items are namespaced under:

```text
data/pilot/<run_id>/content/
data/pilot/<run_id>/attempts/
data/pilot/<run_id>/rendered/
```

Planning creates no files in those output directories and makes no network
calls. Candidate-attempt paths are separate from the one final accepted-content
path. The low-level fresh-run handoff refuses every output collision rather
than inferring resume state from files alone. The generation runner implements
verified resume from its schema-validated, hash-bound state and session
artifacts. After an attempt has been released, its rejected candidate may exist
while the next attempt is released; current/future candidates and all final
content/derivation paths must still be empty.

The run directory itself is also the planner's exclusive publication claim.
Two preflight processes targeting the same brand-new run ID cannot interleave
artifact writes. A crashed writer leaves an intentionally incomplete directory
that must be inspected and explicitly removed before that run ID can be reused.

The manifest is private. It includes topic/content-spec IDs, provider
metadata, source hashes, the contiguous preplanned attempt schedule, candidate
paths, one accepted-content destination, and four derived container records
(including their pair IDs).
None of that private record is automatically merged into a request.

## Hash model

The pipeline separates:

- `request_sha256`: exact canonical request object;
- `provider_profile_sha256`: the complete validated provider profile;
- `provider_execution_sha256`: only provider fields capable of changing
  scientific generation, excluding prices and concurrency;
- `generation_execution_sha256`: request plus
  `provider_execution_sha256`;
- `derivation_sha256`: renderer, localized container catalog, and two fixed
  renderer records;
- request, generation, and derivation bundle hashes;
- manifest, resolved-config, report, notification, approval, and compiled-spec
  hashes; and
- `plan_identity_sha256`, the deterministic scientific/storage identity that
  excludes publication time; and
- `plan_sha256`, the complete published plan including `created_at` and
  excluding only its own hash field.

This division makes wrapper blindness testable. A wrapper-only change moves the
derivation and plan hashes but leaves the content request unchanged. Runtime
metadata records the Python executable hash, implementation/cache tag, platform,
and complete installed distribution inventory; the lock additionally pins
project metadata, dependency lock, and all pipeline source modules.

## Commands

From the repository root, prepare the isolated environment:

```powershell
uv sync --project contrastive --locked
```

Validate the source catalogs:

```powershell
uv run --project contrastive --locked contrastive-data catalog
```

Create a complete bilingual development plan:

```powershell
uv run --project contrastive --locked contrastive-data preflight `
  --provider-config tests/fixtures/provider.yaml `
  --run-id contrastive_preflight_dev
```

Create a development slice:

```powershell
uv run --project contrastive --locked contrastive-data preflight `
  --provider-config tests/fixtures/provider.yaml `
  --run-id contrastive_slice `
  --topic t001 `
  --content-spec content_p01 `
  --language ja
```

Verify an existing run before handoff:

```powershell
uv run --project contrastive --locked contrastive-data verify `
  runs/contrastive_preflight_dev
```

After semantic approval and both empirical calibration reports pass against the
current catalog hashes, add `--production`. Production rejects incomplete
selection, one-language plans, missing credentials, pending semantic approval,
and pending, failed, or stale calibration evidence.

Run all tests:

```powershell
uv run --project contrastive --locked python -m unittest discover `
  -s contrastive/tests -v
```

Planning and verification can also be run directly with the project-local
virtual environment created by `uv sync`:

```powershell
contrastive/.venv/Scripts/python.exe -m unittest discover `
  -s contrastive/tests -v
```

## Acceptance evidence

The current v2-only active surface was checked on 2026-07-28 in the locked
local environment and made no provider/model calls:

- `uv sync --locked` resolved and checked all eight locked packages;
- the automated suites cover pre-generation, generation, calibration,
  post-generation, catalog, and localization behavior, including matched-pair
  negative mutations, recursive schema-surface checks, and concurrent
  divergent-writer controls;
- full-grid tests reconstructed 800 logical jobs, 2,400 exact requests, and
  3,200 derivations;
- the active import graph, CLI, schemas, prompts, configs, and tests contain no
  retired per-container generation pathway; and
- production planning correctly remains blocked by the two pending empirical
  calibration reports; semantic approval, credentials, and both cost gates
  remain independently enforced.

The automated suite covers:

- compilation of exactly ten bilingual shared content specs;
- absence of template/private fields from the content projection;
- exact Japanese semantic-surface language checks;
- cue, language, and private-leak mutation failures;
- wrapper mutations leaving request hashes unchanged while moving derivation
  hashes;
- full 800-job/2,400-attempt/3,200-derivation expansion;
- balanced language, domain, content-spec, and class counts;
- selection-order and run-ID generation-identity invariance;
- cost and context checks;
- provider placeholder, embedded-secret, non-finite-number, and hidden-sampling-
  retry rejection;
- regeneration-prefix stability and attempt-seed uniqueness;
- production/development gate behavior;
- lock mutation detection;
- immutable write idempotence;
- whole-run verification;
- request tamper detection, including after handoff preparation; and
- recursive schema-surface leakage and unsupported-reference rejection;
- synchronized divergent-writer exclusion for a new run ID; and
- byte preservation of all existing topic and container sources.

## Boundary and next stage

This pre-generation stage ends with verified generation jobs. The implemented
generation runner:

1. call `prepare_generation_handoff(run_dir)` once before its first API call;
2. select a private manifest job for scheduling;
3. release attempt zero with `submission_for(job_id, 0)`;
4. route with `adapter_profile` but give only `request` to the model;
5. retry the same envelope only for a true transport failure;
6. advance to the next preplanned attempt after a completed malformed or
   content-rejected response, stopping permanently after success or exhaustion;
7. prohibits arbitrary provider payload overrides and audits any
   provider-generated textual schema/tool wrapper; and
8. records the accepted attempt ID/index/request hash with the structured
   content payload, then renders that one payload through both private
   derivations.

The runner publishes candidate, accepted-content, and derived-container
artifacts with exclusive creation or an atomic no-clobber protocol, never an
ordinary overwriting write. It must reject symlink, junction, or other reparse
aliases in publication paths. Handoff collision checks are a fail-fast guard,
not a filesystem reservation.

The runner implements post-response schema/QC, blind cross-provider extraction
and comparison, the persisted attempt state machine, accepted-payload
provenance, and local rendering. Verification covers both pair-shared bodies
with the purpose frame masked, so acceptance does not condition on either
class. A QC-passing author candidate is durably stored as
`verification_pending` before verifier calls. Completed body checks are saved
individually; transport exhaustion preserves that progress and resumes
verification without consuming a new author attempt. A completed verifier
rejection consumes the attempt. The transport-only circuit breaker admits a
half-open recovery call after its configured cooldown.

Offline corpus-level reconstruction, disjointness checks, grouped folds,
controls, and calibration handoffs belong to `postprocess`. Old run directories
are not portable snapshots: they hash-reference source files in the checkout.
To verify a historical run after repository evolution, restore its frozen
checkout and locked runtime environment.

The one-release cursor is process-local to a handoff object; it is not a
cross-process lease. The generation runner must acquire a single-owner,
persisted run lease before preparing a handoff so that two processes cannot
release the same attempt concurrently. The handoff parses each control-plane
file from the same byte snapshot it hashes and rechecks those bytes before each
release; the persistent lease and no-clobber publication protocol close the
remaining cross-process race at execution time.

Before any provider credential check or model call, the runner also binds the
complete scientific verification policy to
`runs/<run_id_a>/generation_policy.json` under the pair lease. The artifact
records both verified pre-generation plan hashes, normalized local run paths,
the production flag, and the complete verification configuration, including
the generated Japanese surface thresholds. Worker count and lease timeout are
excluded as operational controls.

On resume, the active policy must equal the bound artifact exactly. A mismatch
hard-stops before provider calls. Existing generation state without a policy
artifact is never backfilled; those legacy run IDs must not be resumed. Each
Japanese deterministic-QC result records the applied thresholds, and
`generation_result.json` records the policy path and semantic hash.
