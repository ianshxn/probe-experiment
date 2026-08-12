# English–Japanese localization review

**Status:** assumed approved under the study owner's explicit instruction; no separate native-reviewer identity supplied
**Last consolidated:** 2026-07-28

This is the authoritative localization record for `topics.ja.yaml` and the 40
Japanese container views. It replaces the former topic translation record,
container translation record, container diagnostic, and blueprint-translation
philosophy documents.

## 1. Approval status

The Japanese topic and container views were review-gated drafts. Automated
checks establish structure, lineage, language-surface constraints, and request
safety. They do not by themselves establish complete semantic equivalence or
native naturalness — that judgment required the human review below.

Production pre-generation was blocked until:

1. a researcher approved the paired scientific meaning;
2. a native Japanese reviewer approved equivalence and naturalness;
3. both identities and an ISO-8601 approval timestamp were recorded in
   `prompts/pre_generation_semantic_approval.json`; and
4. the integrity lock and full test suite were refreshed successfully.

For this implementation pass, the study owner explicitly instructed the
pipeline to assume human semantic approval. The approval record and lock
therefore say `approved`, and the reviewer field transparently records that no
separate native-reviewer identity was supplied. If a named attestation is later
required for preregistration, corpus freeze, or scientific release, replace the
assumption and bind the new record to the then-current hashes.

Candidate-output approval, source-view approval, and a confirmatory freeze are
different claims: this approval covers source-view semantic equivalence only.
Any later change to reviewed prose, catalog projections, or model-visible
projection code invalidates it and reopens this record to `pending` — passing
automated tests must not silently re-promote a changed draft back to approved.

## 2. Localization doctrine

English is the canonical semantic authority. Japanese is a paired semantic view
of the same experimental object, not an independently authored condition and not
a runtime translation of an English stimulus.

The governing rules are:

1. Preserve permanent identity, experimental class, pairing, and lineage.
2. Localize every model-visible instruction into natural Japanese.
3. Preserve logical and pragmatic force rather than English word order.
4. Preserve facts, boundaries, units, anchors, slot order, option semantics,
   answer-key behavior, and response restrictions.
5. Add no culturally plausible pressure, hierarchy, urgency, check, or
   consequence absent from the English source.
6. Keep English and Japanese generator sessions strictly monolingual.
7. Use separate mechanical, semantic, and human-review layers for the claims
   each can legitimately establish.
8. Treat an English semantic edit as invalidating the paired Japanese lineage.
9. Do not describe a Japanese view as frozen until review is explicitly
   recorded.

Natural Japanese is an experimental control, not a cosmetic preference.
Translated-sounding prose, altered politeness, or a changed response restriction
could become a language-specific cue.

## 3. Topic pair contract

`topics.yaml` is canonical. `topics.ja.yaml` carries the paired Japanese display
view.

Invariant fields include:

- schema and catalog versions;
- legacy-source and retained-grid metadata;
- every `topic_id`, `topic_label`, and `domain`;
- domain identifiers and ordering; and
- the paired-view metadata block.

Localized fields include:

- the catalog description;
- eight domain display names;
- 40 topic display names; and
- any future language-facing generation guidance.

Stable English machine labels remain unchanged for joins, seeds, balance checks,
and dataset identity. Generator requests use localized display text.

The Japanese metadata pins a canonical SHA-256 of the parsed English catalog
with its top-level `i18n` block removed. Semantic English changes invalidate the
pin; comments and formatting do not.

Mechanical topic validation checks:

- both views against `schemas/topics.schema.json`;
- identical schema shape and machine identity;
- domain-set and ordering equality;
- current catalog version and canonical English hash;
- retained-grid integrity and balance; and
- absence of full-width digits in the Japanese view.

Native review must examine all eight domain names and 40 topic names in context.
Priority scope distinctions include `bug_triage`, `schedule_arithmetic`,
`onboarding_docs`, and `personal_logistics`.

## 4. Container pair contract

Each `containers/cNN.yaml` has one `containers/cNN.ja.yaml` sibling. The two
files represent one container identity.

Strictly invariant fields include:

- `container_id`, `pair_id`, `content_spec_id`, and `class`;
- the hidden English-canonical `design_notes`; and
- paired-view lineage metadata.

Japanese-localized model or subject-facing fields include:

- `display_name`;
- `register` and `role_structure`;
- slot names and template placeholders;
- item-number and response-contract vocabulary;
- template prose and role markers;
- option and answer-key instructions; and
- `model_guidance`.

These localized control values prevent English vocabulary from entering a
Japanese generator request. Machine identity remains English-canonical outside
the model-visible projection.

Each pair pins a canonical SHA-256 of the parsed English container with `i18n`
removed and targets the current container-catalog version. Mechanical validation
checks:

- both files against `schemas/container.schema.json`;
- matching pair shape and permanent identity;
- validated English-to-Japanese mappings for structural fields;
- matching slot order recovered from the actual template;
- item-number and response-contract parity;
- current catalog version and English semantic hash;
- absence of full-width digits; and
- balanced register, role, and option-format structure across classes.

The pre-generation pipeline additionally compiles the four containers assigned
to each content specification (two matched eval/deploy pairs) into one shared,
template-blind content specification. Japanese requests are
audited after complete rendering: semantic fields may contain neither Latin
characters nor full-width digits. Private class, template, identifier, and design
metadata is rejected at the request boundary.

## 5. Preserving the contrast

Evaluation containers must retain their intended assessment framing. Deployment
containers must not acquire evaluation vocabulary, stronger answer constraints,
or keyed-distractor phrasing.

The frozen Japanese cue screen includes terms such as:

- `テスト`
- `試験`
- `評価`
- `ベンチマーク`
- `採点`
- `監査`
- `検証`
- `査定`

Zero such hits are allowed in deployment wrappers or generator requests.
Evaluation containers may intentionally contain evaluation framing; that is the
construct rather than leakage.

Every option-bearing deployment container must present four genuine
alternatives with the same answer semantics as its eval twin. Only the
purpose-frame construct may differ within a pair. Whether generated
alternatives remain plausible is a generation-time verification property and
cannot be proven from an empty container.

## 6. Completed review evidence

Mechanical catalog and request checks cover:

- all 40 topic pairs;
- all 20 container pairs;
- all ten bilingual shared content specifications;
- all 800 logical full-grid jobs; and
- all 1,200 Japanese attempt requests.

The active v2-only suites check schema validity, pair invariants, hashes, slot
and placeholder contracts, template blindness, recursively nested Japanese
schema surfaces, localized verification schemas, cue screening, exclusive run
publication, immutable planning, and protected-source byte preservation.

An LLM self-review of the container pairs found one pragmatic issue and corrected
it:

- **c16:** the first draft opened with `ちょっと聞きたいんだけど`, which read as
  a plain casual question and weakened the eval container’s “quick check”
  framing. It was changed to `ちょっと確認したいんだけど`, preserving the check
  function while remaining casual.

That self-review is useful diagnostic evidence, but it was not independent blind
extraction and was not itself native-speaker approval. Independent researcher
and native-speaker review was completed for the earlier integrity binding. The
current cleaned binding was reviewed again by Codex for the demo override. That
review used a Japanese-only pragmatic extraction before English comparison,
covered all topic/domain labels and wrapper pairs, and corrected translated-looking
punctuation in c02, c10, c11, c16, c17, and c18 plus the c14 wording for “one
attempt.” It remains automated diagnostic evidence, not renewed human/native
approval, as described in section 1.

## 7. Human-review priorities

The native reviewer should prioritize:

- every implicit frame in both classes;
- every option-bearing and constrained-output pair;
- casual and terse registers;
- each complete frame-body combination, especially where institutional
  purpose framing meets conversational body prose;
- naturalness and translated-sounding prose across deployment items; and
- the topic-scope distinctions listed in section 3.

For every container, reviewers should compare:

- evaluation or deployment framing;
- response restrictions and answer-key behavior;
- professional, casual, or terse register;
- system/user relationship;
- option semantics;
- degree of explicitness and certainty;
- absence of new pressure, urgency, hierarchy, or consequences;
- internal coherence of each complete frame-body prompt; and
- absence of a stronger body cue that overwhelms the purpose frame.

## 8. Required blind and generated-output checks

Before semantic approval:

1. Give an independent reviewer or extractor only the Japanese topic/container
   model view, without English source text or expected labels.
2. Extract pragmatic function, response restrictions, register, option
   semantics, and role structure.
3. Compare that extraction against the English intent in a separate pass.
4. Resolve every material difference and rerun the mechanical suite.
5. Obtain native-speaker review of the complete paired views.

After real generation:

1. inspect Japanese output rendered through every container family;
2. verify that eval/deploy framing survives filled slots;
3. audit deployment naturalness, especially c02, c04, and c05;
4. run the frozen cue, numeral, length/register, and disjointness gates; and
5. record regeneration and acceptance provenance.

## 9. Approval binding

Human approval is bound to three frozen surfaces:

1. bilingual prompt and scaffolding prose;
2. the canonical topic/content-spec semantic projection; and
3. every non-circular locked implementation/runtime input that can affect
   request construction or enforcement.

Any reviewed semantic or implementation change invalidates the corresponding
hash. The refresh helper refuses to carry an existing approval across such a
change. See
[`../PRE_GENERATION_PIPELINE.md`](../PRE_GENERATION_PIPELINE.md#scaffold-lock-and-semantic-approval)
for the complete operational procedure.
