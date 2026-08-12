# Test migration bundle for a redesigned contrastive-set experiment

## Purpose of this folder

This folder is a non-authoritative migration staging area for a possible new,
sibling repository. It collects source files from `contrastive/` that may be
reused, adapted, or consulted while designing a new evaluation-awareness probe
experiment.

No copied source file has been edited. The originals remain in the current
repository, and the copies retain their paths beneath `from_contrastive/`.

Source provenance:

- source repository: `C:\Users\ishen\LANG-EA-PILOT`
- source subtree: `contrastive/`
- source branch: `ian-branch`
- source commit: `df858754a99817a6242979c07b97618883893c98`
- migration prepared: 2026-08-10

This bundle is not itself a frozen protocol, production dataset, or independent
confirmation bank.

## Conversation summary and current decision context

### Original contrastive-set objective

The existing contrastive set was designed to fit, separately for each subject
model, a linear direction distinguishing evaluation-purpose prompts from
deployment-purpose prompts. The deployment condition is the reference pole; the
design does not identify two independent evaluation and deployment axes.

The current architecture uses content-matched evaluation/deployment twins. A
generated payload is shared within a matched pair, while a renderer inserts a
purpose frame. English and Japanese are both included. The design contains:

- 40 English and 40 Japanese container files;
- 20 purpose-frame pairs across five held-out frame families;
- implicit and explicit purpose frames;
- external and institutional surfaces;
- noun-only, verb-only, and both-locus cue constructions;
- 40 topics across eight domains; and
- grouped probe evaluation intended to keep shared payloads together.

The `demo16_v2` development corpus contains 300 accepted generation jobs, 1,200
rendered fitting items, and 600 matched pairs. The configured full production
grid contains 800 planned jobs and would emit 3,200 items under the current
four-items-per-job postprocessing rule.

### Probe findings

The stand-in probes reliably separate the controlled evaluation and deployment
families, but retain measurable lexical sensitivity. All three prospective 8B
confirmation attempts were rejected under their predeclared cue-effect bounds.
The latest constrained recipe preserved purpose ordering but failed one fresh
cue-invariance cell: Meta English implicit-cue effects exceeded the allowed
bound.

The strongest supported claim is therefore narrow:

> Purpose is linearly decodable in the controlled sets, but the evidence does
> not establish a cue-invariant purpose-attribution direction or naturalistic
> transfer.

No production probe recipe is adopted. The existing naturalistic-transfer set
has not been inspected and remains sealed. The three completed confirmation
banks are spent and may be used only as development evidence.

### Proposed redesign from those findings

For the stronger purpose-attribution claim, purpose and cue polarity should be
crossed inside the fitting corpus rather than leaving crossed cues in a small
validation add-on. Within a payload block and cue family/context, the renderer
should materialize the complete 2 x 2:

| Intended purpose | Evaluation-polarity cue | Deployment-polarity cue |
|---|---|---|
| Evaluation | aligned | crossed |
| Deployment | crossed | aligned |

The payload must be identical across the four cells. Each cue term should occur
equally often with both purposes, preventing cue polarity from serving as a
class shortcut.

The payload generator should remain class-blind. It should not author separate
evaluation and deployment content, because class-conditioned payloads would
discard one of the current design's strongest controls.

### Relationship to arXiv:2603.19426v1

The paper *Is Evaluation Awareness Just Format Sensitivity? Limitations of
Probe-Based Evidence under Controlled Prompt Structure* studies a different
confound. It crosses evaluation/deployment context with benchmark/casual format.
Its paired training substantially reduces benchmark-format false positives, but
several cells come from different source datasets, so source, task type, and
content are not fully matched.

The proposed local redesign is stronger for identifying cue-independent purpose
effects because it uses identical payloads within complete blocks. It is not a
complete replacement for the paper's format control: the two designs address
different nuisance axes.

The strongest combined experiment would cross:

```text
purpose x cue polarity x format
```

This is a 2 x 2 x 2 design. It would preserve identical semantic payloads where
possible while independently varying purpose, cue polarity, and benchmark-like
versus casual/free-form structure.

### Expected size of the combined change

Integrating format alone into the current instrument is a medium-to-large
architectural change. Combining it with the purpose-by-cue redesign is a large
change, but not a total rewrite: provider access, run management, activation
extraction, grouped statistics, and parts of verification remain reusable.

Assuming the current job grid and complete factorial rendering:

| Design | Rendered items per current job | 300-job equivalent | 800-job equivalent |
|---|---:|---:|---:|
| Current architecture | 4 | 1,200 | 3,200 |
| Add purpose x format | 8 | 2,400 | 6,400 |
| Purpose x cue polarity x format | 16 | 4,800 | 12,800 |

Payload-authoring calls need not grow at the same rate if a single canonical
payload can support every rendering. Activation extraction, storage, structural
QC, and comprehension checks would grow roughly with the rendered-item count.

The main implementation difficulty is creating a canonical semantic payload
that can be rendered faithfully in both benchmark and casual formats. The
current system has content-specification-dependent payloads: multiple-choice
and free-response forms are not currently alternate renderings of one canonical
task.

### Repository decision

A new sibling repository is recommended if this becomes a distinct experiment,
preregistration, or paper. The current repository contains historical waivers,
hash-bound artifacts, spent confirmation banks, and conclusions tied to the old
estimand. A separate repository gives the new experiment an unambiguous evidence
boundary.

The new repository should retain this source commit as provenance and should
reuse infrastructure deliberately rather than copying all historical runs and
outputs. If the new instrument later replaces the original production design,
shared infrastructure can be extracted into a package or merged back.

### Hashing policy discussed

Hashing and hash enforcement should be treated separately:

- during development, hashes describe what happened;
- during confirmation, hashes constrain what is allowed to happen.

The new project should compute observational hashes from the beginning for raw
inputs, exact model-visible text, model responses, rendered items, model and
tokenizer revisions, and activation-to-item associations. Development should
not use broad hard-coded locks that make every legitimate edit fail globally.
Instead, each run should write a new manifest and reject only internally
inconsistent combinations, such as activations paired with different stimulus
text.

Strict expected-hash enforcement should begin only after the estimand, schema,
renderer, split rules, probe recipe, gates, and independent confirmation plan
are frozen.

## What was collected

All files below are unchanged copies under `from_contrastive/`.

### Portable or mostly portable infrastructure

- `src/contrastive_generation/utils.py`
- `src/contrastive_generation/providers.py`
- `src/contrastive_generation/lease.py`
- `src/contrastive_generation/allocation.py`
- `src/contrastive_generation/planning.py`
- `src/contrastive_generation/publication.py`
- provider and model-request schemas
- `configs/providers/provider.example.yaml`
- project packaging metadata

These provide atomic artifact writing, structured data loading, provider access,
retry behavior, run leasing, deterministic allocation/seeding, and generic
schema patterns. Some path constants and experiment-specific assumptions still
need removal.

### Architecture source to adapt

The full `src/contrastive_generation/` package, schemas, and tests were included
because even tightly coupled modules contain useful implementations and
regression patterns. In particular:

- `catalog.py` and `content_specs.py` show bilingual catalog and payload checks;
- `runner.py` shows provider orchestration, retries, and immutable state bundles;
- `verification.py` shows structured extraction and comparison verification;
- `post_generation.py` shows shared-payload validation and grouped exports;
- `pre_generation.py` and `preflight.py` show planning and integrity checks; and
- the tests capture failure cases that should inform the replacement system.

These files should not be treated as the new architecture unchanged.

### Topics and container source material

The bundle includes:

- `topics.yaml` and `topics.ja.yaml`;
- `containers.yaml`;
- all 40 English and 40 Japanese container files;
- `purpose_frames.yaml`; and
- their schemas.

Topics are highly reusable. Stable topic IDs, domain assignments, and bilingual
mapping can migrate, but old `legacy_design` metadata should be removed and new
format-compatibility/task-form fields should be added later. Japanese labels and
any new format-specific guidance require review.

Containers are reusable as design ingredients, not as intact experimental
units. Each current container bundles purpose class, purpose wording, cue locus,
frame family, body format, response contract, register, role structure, and
content specification. The new architecture should decompose these into
independent catalogs such as:

```text
canonical_payload_schemas
purpose_frames
cue_variants
format_templates
nuisance_profiles
```

Catalog 2.3.2 frames are development material. They produced the existing probe
findings and contain known lexical/noun-locus weaknesses, so they cannot provide
fresh confirmation evidence.

### Prompt and calibration patterns

The copied generation and verification prompts are references for class-blind
payload generation, renderer-owned framing, and bilingual structured
verification. They must be rewritten for canonical payloads and independent
format variants.

The comprehension-check code and selected scripts are included as adaptable
implementations. Existing calibration reports and reader outputs were not
copied.

### Probe and factorial-control code

The bundle includes selected activation extraction notebooks/scripts, grouped
probe analysis utilities, layer/readout development code, cue-invariant and
cue-constrained estimators, and the builders/evaluators for the three historical
factorial confirmations.

The algorithms and data-shape checks may be reused. The historical recipes are
not adopted production methods, and the confirmation code embeds old paths,
field names, layers, thresholds, and bank-specific assumptions that must be
replaced or explicitly revalidated.

### Design and evidence references

Selected documentation was copied to preserve:

- the estimand and intended claim;
- the control-coverage diagnosis;
- cue-locus and frame diagnostics;
- current claim boundaries;
- the post-generation grouping protocol;
- the three-confirmation interpretation; and
- the prospective crossed-fitting redesign.

For the current repository, numerical metric files and corresponding result
documents remain the authoritative evidence. This migration bundle is a design
handoff, not a substitute numerical archive.

## Deliberate exclusions

The following were not copied:

- `.venv/` and `.pytest_cache/`;
- `runs/`;
- generated or processed `data/`;
- activation outputs under `probe_check/out/`;
- calibration reports and comprehension model outputs;
- real provider profiles;
- `uv.lock`, which should be regenerated for the new project;
- the old pre-generation lock and semantic-approval record;
- old generation configurations containing current waivers;
- existing probe metrics and narrative results as executable inputs;
- the three spent confirmation/control data banks;
- legacy lexical-control data;
- the sealed naturalistic-transfer set; and
- generated development augmentations.

The unadopted `validation/proposed_noun_controls.yaml` was included only as
design source material. It must not be mistaken for a frozen or validated
control family.

## Scientific status and constraints another LLM must preserve

1. Do not describe any current probe recipe as adopted.
2. Do not reuse the three scored confirmation banks for a new confirmation
   verdict.
3. Do not inspect or score the original naturalistic-transfer set during
   development.
4. Keep payload generation class-blind and share one canonical payload across
   every factorial cell in its block.
5. Keep purpose, cue, and format as separately represented factors.
6. Group splits and bootstrap resampling by whole payload blocks, not individual
   rendered rows.
7. Hold out purpose-frame families and test within-format and cross-format
   generalization.
8. Treat literal class labels and format-only markers as nuisance shortcuts.
9. Estimate construct-adjacent cue effects by term and family, with an
   anti-cancellation rule so opposing large effects cannot disappear into a
   small average.
10. Run English and Japanese gates separately; a pooled result cannot rescue a
    failing language.
11. Obtain new Japanese review for purpose wording, cue contexts, format
    realizations, and cross-language semantic equivalence.
12. Freeze the complete protocol before authoring or scoring a new independent
    confirmation bank.

## Suggested new architecture

A clean implementation should make the factorial block explicit:

```text
canonical payload
  -> purpose variant
  -> cue variant
  -> format variant
  -> rendered item
```

Minimum recommended identifiers and metadata:

- `payload_block_id` and `matched_group_id`;
- language and bilingual correspondence ID;
- intended purpose;
- purpose-frame family and variant ID;
- cue family, ontology category, term-pair ID, and polarity;
- cue context and position;
- format family and variant ID;
- alignment/crossing indicators;
- topic, task form, and source IDs;
- generator, verifier, source, and rendered-text hashes; and
- split/group identifiers used by fitting and bootstrap procedures.

Do not reproduce the old design by creating a separate monolithic container for
every combination. A compositional renderer makes completeness and balance
easier to validate and prevents catalog size from growing combinatorially.

## Recommended development sequence

1. Initialize a clean sibling repository and record the source commit above.
2. Copy or package the portable infrastructure from this bundle.
3. Define the estimand and cue ontology before implementing the schema.
4. Define a canonical payload representation that supports matched benchmark
   and casual renderings.
5. Decompose topics, purpose frames, cues, format templates, and nuisance
   profiles into independent catalogs.
6. Implement complete-block planning and deterministic rendering.
7. Implement development-mode manifests with observational hashes but without
   broad expected-hash locks.
8. Port grouped activation extraction and probe analysis against the new
   metadata contract.
9. Use only spent/development material to choose the relative-depth grid,
   readout, estimator, regularization, and term-aware gates.
10. Freeze one protocol, enable strict hash enforcement, then author a new,
    disjoint confirmation bank.
11. Open a compatible naturalistic-transfer set only after controlled
    confirmation passes and only under the frozen one-shot rule.

## Directory layout

```text
test-migration/
  README.md                 # this migration and LLM handoff
  from_contrastive/         # unchanged copies preserving original paths
```

Before promoting anything from this staging folder into a new repository,
classify it as one of:

- reusable infrastructure;
- architecture source requiring adaptation;
- development-only scientific material; or
- historical documentation.

Nothing in this folder should be assumed production-ready merely because it was
copied.
