# Probe readiness

## Current status

The checked-in catalog is version 2.3.2 and is structurally ready to generate
fitting stimuli and assemble them for subject-model activation collection. It
does **not** yet have a production corpus, subject-model activations, or empirical
production-probe results. The 8B work comprises development analyses and three
prospective stand-in confirmations; it is binding for stand-in recipe adoption,
not gate evidence on the declared subject models.

The older `slice_qwen` / `slice_gemini` outputs were generated with the
superseded container catalog. The current postprocessor rejects that stale
provenance; those outputs cannot be presented as evidence for this design.

Frame-only human calibration and complete-prompt held-out-model comprehension
remain empirically unresolved for catalog 2.3.2. Production preflight currently
proceeds only through recorded waivers; a passing preflight must not be described
as passing these empirical gates.

The full explicit-only development forecast is complete. Eleven of twenty cells
pass every controlled representation gate on both 8B stand-ins, and seven also
pass threshold-free controls. The first prospective full-format confirmation
then passed every purpose-ordering cell but failed all eight lexical-cue-effect
bounds, rejecting the unprojected explicit-only recipe. That bank is now spent.

A block-cross-fitted comparison on the spent first bank selected an SVD90
nuisance projection at the already frozen `mean` layer 6 cell. On the second,
text-disjoint bank, every purpose-ordering and implicit-cue gate passed, but the
explicit-lexeme cue effect exceeded 0.25 fitting SD in both languages on both
models (0.272â€“0.417). The joint verdict rejects SVD90 for production adoption.
The first two confirmation banks are spent. Naturalistic transfer remains sealed.
See
[`../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`](../probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md).

The stronger-claim redesign remains development-only and currently has no
confirmation candidate. After the initial four recipes selected none, the
five-context extraction supported two predeclared constrained rounds.
`balanced_penalty_256` alone passed every context and core gate on both models.
The completed third confirmation rejected it on Meta English implicit cues
(0.3566 SD versus 0.25), although every purpose gate passed. The bank is spent,
no production recipe is adopted, and naturalistic transfer remains sealed. See
[`../probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md`](../probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md).
The combined interpretation and requirements for any future redesign are
archived in
[`PROBE_REDESIGN_ARCHIVE_2026_08_06.md`](PROBE_REDESIGN_ARCHIVE_2026_08_06.md).

## Threats closed in the checked-in design

### Purpose is the sole matched difference

There are 20 eval/deploy pairs (40 logical containers, stored as 40 English and
40 Japanese localized files). Within each pair:

- the generated payload is byte-identical;
- body template, response contract, answer format, speaker presence, prior
  context, scaffolding, topic, and content specification are shared;
- the purpose-frame attribution and text are the only class-varying fields; and
- strict pair, catalog, rendering, and artifact hashes are rechecked after
  generation.

Numbering and XML are absent from the fitting wrappers. First-person framing is
absent from the short purpose frames. These nuisances appear only in the frozen
naturalistic transfer set, where they test out-of-distribution transfer rather
than label the fitting class.

### Nuisances are balanced

The frozen 20-pair manifest balances explicitness, surface, output constraint,
speaker presence, and prior context 10/10. Every pairwise binary cell has five
pairs. Constrained items are split five/five between bare-answer and
bounded-explanation contracts; unconstrained items use neither.

Raw eval/deploy frame lengths are gated per pair, overall, and within
explicitness and surface strata in both languages.

### The contrast structure is declared and matched across languages

Every pair declares a `cue_locus` — `noun_only`, `verb_only`, or `both` —
recording which element carries the eval/deploy contrast. It is declared once
per pair rather than once per language, so the English and Japanese members
cannot drift into different contrast structures. That drift is the failure the
field was introduced to close: catalog 2.1.0 held the agent noun constant in
four English implicit pairs whose Japanese counterparts varied both the noun
and the verb, which gave Japanese systematically more redundant cueing than
English on the primary comparison.

Across the ten implicit pairs the locus is balanced four `both` / three
`verb_only` / three `noun_only`, no frame family carries the same locus twice,
each locus spans at least three families, and each surface stratum contains all
three loci. The explicit stratum is uniformly `noun_only` by construction.

Three machine gates enforce this in `_validate_cue_locus`. A pair declaring a
single locus must differ in exactly one contiguous span; a pair declaring
`both` must differ in at least two; and no span may recur in more than two of
the ten implicit pairs of a class in either language. Spans are computed
morphology-free — whitespace tokens for English, characters for Japanese — so
the gate measures the surface material a probe could key on rather than part of
speech.

The span-frequency cap is what makes the implicit stratum meaningfully
implicit. Under 2.1.0 the Japanese deploy verb 活用 appeared in all ten implicit
deploy frames and nowhere else, making it a perfect single-token class
indicator that no leave-one-frame-family-out fold could remove; 確認 appeared in
seven eval frames and `apply` in four English deploy frames. Structural gates
cannot certify naturalness, only that no single cue separates the classes.

### Frame memorization is testable

The 20 frame pairs contain 40 globally unique purpose-frame texts in English
and 40 in Japanese. They are assigned to five frozen frame families. Each
family has exactly one pair in every explicitness-by-surface cell.

Each content specification feeds two wrapper pairs in different frame
families. Therefore leave-one-family-out validation does not remove an entire
content specification from training, and frame family is not identical to
content-spec group.

**That holds only when both strata are pooled.** The content specification's
second wrapper pair lives in the explicit stratum, so within the implicit
stratum alone -- which is exactly how the contract requires recoverability to be
reported -- frame family and content specification are perfectly confounded.
Verified on extracted metadata (2026-08-04): 300 implicit payloads each in
exactly one frame family, and 10 content specs each in exactly one frame family,
so holding out `ff03` also removes `content_p07` and `content_p10` entirely.
This makes the implicit-only fold conservative rather than optimistic, but an
implicit-only fold result cannot be attributed to frame-family generalisation
specifically. The same caveat applies symmetrically to explicit-only fitting:
within either stratum, held frame family and held content specification move
together. Either state this in the writeup or cross the design so each stratum's
frame families carry content from more than one specification.

The postprocessor writes `folds.json` with five leave-one-frame-family-out
folds. Random item splitting is explicitly prohibited.

### Lexical shortcuts are testable

`validation/lexical_controls.yaml` contains 48 frozen bilingual legacy controls
in two cue families:

- `explicit_lexeme` (32 items) — neutral mention-only uses of the
  evaluation/deployment vocabulary, and crossed distractors in which that
  lexical cue and the intended purpose point in opposite directions; and
- `implicit_cue` (16 items) — the same design applied to the scrutiny-versus-use
  verbs the implicit frames actually carry (`check`/`inspect`/`vet` against
  `apply`/`use`/`employ`; 確認/点検/精査 against 活用/利用/用い), so that a probe
  riding on the implicit lexicon is detected rather than assumed absent.

Each control declares its `cue_term`; the validator requires that term to be
present, the counterpart's term to be absent, and every `implicit_cue` item to
contain no explicit evaluation/deployment lexeme at all.

The implicit-cue family also probes the scrutiny confound directly. Its crossed
items pair a deployment purpose with an inspection verb ("a teammate will check
this response before it goes to the customer") against an evaluation purpose
with a use verb, separating "my output will be judged" from
"my output is being evaluated".

These controls are exported separately and remain mandatory diagnostics. Their
bare-prompt format sits −1.68 to +6.01 fitting SD from the framed payloads, so
absolute zero-threshold accuracy cannot be the prospective cross-format gate.

`validation/format_matched_controls.yaml` is the immutable pre-run specification
for the first full-format confirmation:
16 bilingual payload blocks, eight per surface, each crossed over two cue
families, two intended purposes, and two lexical-cue polarities. Its 256 rendered
prompts match the frame-plus-payload structure and observed length range. Cue
terms are isolated to a neutral metadata line; novel implicit terms do not reuse
legacy cue terms. The validator enforces the complete factorial, prompt lengths,
freshness, and disjointness. These controls were frozen before extraction and
could not select the fitting recipe, layer, position, threshold, or calibration.
They have since been scored and are spent development data. The two later banks
are likewise spent; see
[`PROBE_REDESIGN_ARCHIVE_2026_08_06.md`](PROBE_REDESIGN_ARCHIVE_2026_08_06.md).

### Naturalistic transfer is testable

`validation/naturalistic_transfer.yaml` contains 40 frozen bilingual prompts:
one eval and one deployment item for each of ten scenario families and each
language. The set covers first-person, numbered, XML, bundled-evaluation, and
ordinary-workflow formats, plus chat transcripts, email, forms, policy headers,
and incident contexts.

The set is exported separately and may be scored once only after the probe and
all analytical choices are frozen.

### Grouped provenance is available

Every probe-fitting item carries:

- shared payload hash and job ID;
- pair and container IDs;
- frame family and holdout fold;
- topic, domain, and content-spec IDs;
- language, class, explicitness, and surface;
- generator and verifier roles; and
- rendered artifact path and hash.

`matched_pairs.jsonl` records the eval/deploy twins.
`language_calibration_groups.jsonl` records corresponding EN/JA design cells.
`dataset_manifest.json` freezes every emitted artifact.

The postprocessor also writes a blinded, deterministic complete-prompt
comprehension sample and private key. The separate frame-calibration command
writes an immutable blinded packet containing all 80 localized frame sentences.
Both result validators enforce the preregistered gates and model/rater coverage.
The human validator also requires a new confirmatory-cohort record with
language-qualification and blinding attestations. The model validator requires
both languages, every pair and class, at least one prompt per cell, and at
least two mutually lineage-disjoint calibration models.

## What remains empirical

The pipeline cannot establish probe validity without hidden states from each
subject model, but subject-model extraction is not the next step. All three 8B
confirmation candidates were rejected. The remaining sequence is:

1. On spent stand-in data only, define the prospective cue ontology,
   term-aware or hierarchical cue estimand, anti-cancellation gate,
   relative-depth grid, and joint layer/estimator selection rule.
2. If the stronger claim is retained, implement a fitting corpus that crosses
   intended purpose with cue polarity inside payload blocks and validates
   balance by language, surface, cue family, cue term, and context.
3. Freeze the estimator, grouping, thresholds, and one confirmation cell before
   authoring another text-, payload-, term-, and context-disjoint control bank.
4. Run that bank once on both 8B stand-ins. A failure rejects the candidate; a
   pass only authorizes the still-sealed naturalistic transfer gate.
5. Only after controlled and naturalistic stand-in qualification, freeze a
   relative-depth mapping for each subject architecture without using
   main-study outcomes.
6. After the Stage-6 prompt bank freezes, run exact and near-duplicate
   disjointness against probe-fitting text before subject-model activation
   choices or scoring.
7. Report every leave-one-frame-family-out fold, implicit stratum, surface,
   language, cue term, and grouped uncertainty result rather than only pooled
   performance.
8. Calibrate EN/JA scores within subject model using training data only and
   report language-specific as well as calibrated pooled results.

The postprocessor freezes these requirements in
`probe_evaluation_contract.json`, including numerical success criteria. It
deliberately does not invent numerical results before activations exist. Its
current cue gate predates the term-level redesign and must be amended
prospectively rather than silently treated as the future contract.

## Claim boundary

If the empirical gates pass, the defensible claim is:

> The subject model contains a linearly decodable activation direction
> associated with evaluation-versus-deployment purpose attribution, and the
> direction generalizes across held-out framings and to the frozen
> naturalistic contexts.

The design cannot by itself establish conscious evaluation awareness, causal
control of behavior, or separate evaluation and deployment axes. A binary
contrast yields one direction with evaluation and deployment as opposite
poles. Independent axes require a third neutral or unknown-purpose condition.

## Operational readiness

- Semantic approval is waived for the current preflight; catalog 2.3.2 still
  requires current native Japanese review before scientific freeze.
- Frame-only and held-out-model calibration are implemented as executable gates
  but are waived/pending rather than empirically passed.
- All checked-in pre-generation and generation unit tests pass.
- Post-generation integration tests pass with a two-job bilingual mock run.
- No current full generation has been run, and no production fitting recipe is
  frozen.
