# Probe confirmation archive and contrastive-set redesign decision

**Archived 2026-08-06. Current-status authority for the probe program and the
next contrastive-set design stage.**

This document consolidates the three prospective 8B confirmations, the
development work between them, the resulting claim boundary, and the design
requirements for any attempt at the stronger purpose-attribution claim. It is
an archive and a prospective design decision, not evidence that the redesign
has been implemented or validated.

The confirmation-specific result documents and machine-readable metric files
remain numerically authoritative. This document is authoritative for their
combined interpretation, live asset status, and the next permitted actions.

## 1. Binding state

- No production probe recipe is adopted.
- All three prospective factorial control banks have been scored and are spent.
- `balanced_penalty_256`, SVD90, and unprojected `explicit_only` are rejected as
  production recipes under their respective frozen confirmation contracts.
- The frozen naturalistic-transfer set has never been loaded, extracted, or
  scored and must remain sealed.
- No new A100 extraction is currently authorized. A new extraction becomes
  legitimate only after a prospective estimand, cue gate, layer-selection rule,
  fitting design, and independent confirmation plan have been recorded.
- No current full production contrastive corpus exists, and no subject-model
  probe has been trained.

The strongest admissible summary of the stand-in evidence is:

> The tested linear probes separate controlled evaluation-purpose from
> deployment-purpose prompts with high reliability, including within crossed
> cue polarities, but retain measurable and non-generalizing lexical
> sensitivity. The evidence does not establish a cue-invariant purpose
> direction or naturalistic transfer.

## 2. What each stage established

| stage | frozen test and outcome | information retained | consequence |
|---|---|---|---|
| Explicit-only development | Seven shared cells passed the controlled purpose gates and threshold-free controls on both 8B stand-ins. `mean|6` won the recorded purpose-maximin rule at 0.7458. | Purpose information learned from explicit frames transfers to implicit frames over a coherent layer band. Mean pooling is preferable to the frame and last-token alternatives for the tested corpus. | Authorized one prospective test of unprojected explicit-only at the frozen cell; did not adopt the recipe. |
| First confirmation | Unprojected `explicit_only` passed every purpose-ordering gate but failed all eight family-language cue-effect cells, with absolute effects of 0.527-1.580 fitting SD against a 0.25 limit. | The core task and purpose sentences carry a highly decodable contrast. A direction selected only for purpose decodability also responds strongly to lexical cues. | Rejected the unprojected recipe. Bank 1 became spent development data. |
| Cue-invariant development | Block-cross-fitted SVD90 reduced held-block cue effects while retaining the declared core floor and was selected before Bank 2 was scored. | Some purpose and cue variation can be separated linearly on seen cue-difference structure. | Authorized one prospective SVD90 test; did not establish transfer to new cue wording or format. |
| Second confirmation | SVD90 passed every purpose gate and all implicit-cue gates but failed explicit-lexeme effects in both languages on both models. | Removing a nuisance subspace learned from one bank does not reliably remove cue sensitivity under new wording and a new cue-line format. | Rejected SVD90. Bank 2 became spent development data. |
| Reciprocal and five-context development | Initial cue-balanced/direct-penalty recipes failed held-bank invariance. Expanded five-context development eventually selected `balanced_penalty_256`, narrowly passing every declared development gate on both stand-ins. | Cue behavior varies with wording, term, and cue-line context. Directly constraining matched cue swaps is more promising than a single post-hoc subspace, but the margins were narrow. | Authorized one prospective test on a third text- and context-disjoint bank. |
| Third confirmation | Every purpose gate passed with an AUC and interval floor of 1.0. Swallow passed the four aggregate cue cells. Meta failed English implicit cues at 0.3566 SD versus 0.25. | The constrained estimator generalizes to the canonical explicit labels but not reliably to new construct-adjacent implicit terms. Aggregate family means can hide large, opposing term effects. | Rejected `balanced_penalty_256`. Bank 3 became spent. A stronger future protocol requires a new estimand and term-aware gate. |

## 3. What did not fail

The generated task payloads were not implicated by the confirmations. Across
all three banks, purpose ordering remained strong after holding payload,
surface, cue polarity, and language constant. The existing payload architecture
therefore remains the default:

1. the authoring model receives a localized topic and a class-blind content
   specification;
2. it does not receive the purpose label, purpose frame, container, nuisance
   profile, or machine identity;
3. the same accepted payload is reused across matched rendered conditions; and
4. content verification remains class-blind with the purpose wrapper masked.

The redesign must not ask the payload generator to author evaluation and
deployment content separately. Doing so would create class-conditioned content
shortcuts and discard the strongest existing control.

## 4. Mean layer 6: why it was used and what is no longer implied

Mean pooling averages the hidden state over all prompt tokens at one transformer
depth. It was retained because it sees the entire payload, has a larger
effective sample size than the frame read-out, outperformed the last-token
read-out over the tested development grid, and removed a last-position English
`noun_only` artifact.

Layer 6 was selected before fresh controls were scored. Among seven jointly
admissible cells (`last|6` and `mean|6,8,10,12,14,16`), it maximized the minimum
of four purpose-decodability metrics across both models. It was then held fixed
to avoid post-confirmation layer search.

That choice is procedurally valid but not evidence that layer 6 is the unique or
best cue-separated purpose layer:

- its winning margin over `mean|12` and `mean|16` was 0.0091;
- all selection components measured purpose decodability rather than cue
  invariance;
- the selection score correlated +0.91 with the available cue-effect proxy;
- `mean|6` was the worst of the seven candidates on that proxy; and
- later confirmation extractors stored no alternative mean-pooled layers, so
  their counterfactual verdicts are unknown.

Future development must use a preregistered **relative-depth** grid, not transfer
absolute layer 6 to deeper subject architectures. Layer, read-out, and estimator
must be selected jointly on grouped development data. A later confirmation must
again score exactly one frozen cell. Other layers cannot retroactively rescue
any of the three completed confirmations.

## 5. Prospective estimand

The old phrase "lexically cue-invariant" is too broad. Some cue terms are merely
labels or formatting artifacts, while others convey genuine information about
what will happen to the response. Requiring a probe to ignore every synonym of
inspection, scoring, rollout, or practical use can remove the same semantic
features that define the intended construct.

The recommended target is:

> A linear evaluation-versus-deployment purpose direction that generalizes
> across held-out purpose frames and is not dependent on the catalog's literal
> class labels or cue-line formats, with construct-adjacent lexical sensitivity
> estimated and bounded at the term and family levels.

This target does not claim invariance to arbitrary purpose-bearing language.
The cue ontology must distinguish:

1. **literal class labels** such as `evaluation` and `deployment`, treated as
   nuisance shortcuts under a strict invariance gate;
2. **format-only context** such as `Reference keyword`, `Context tag`, or cue
   placement, also treated as nuisance variation;
3. **construct-adjacent semantic terms** such as `inspection`, `measurement`,
   `rollout`, and `practical use`, whose effects are estimated term by term and
   cannot be allowed to reverse purpose ordering; and
4. optional **neutral context terms**, used as calibration controls rather than
   as a third purpose class.

If the study instead requires separate evaluation and deployment axes, a third
neutral or unknown-purpose condition is necessary. That would be a different
estimand and a larger redesign than the binary direction archived here.

## 6. Required generation redesign for the stronger claim

Purpose and cue polarity must be crossed inside the fitting corpus rather than
appearing only in a small validation add-on. Within a payload block and a
declared cue family/context, the renderer must materialize the complete 2 x 2:

| intended purpose | evaluation-polarity cue | deployment-polarity cue |
|---|---|---|
| evaluation | aligned | crossed |
| deployment | crossed | aligned |

The purpose sentence, cue line, and payload remain distinct rendered fields.
The payload is identical across the four cells. Each cue term appears equally
often with both intended purposes, preventing the fitting objective from using
cue polarity as a class shortcut.

A balanced incomplete-block allocation is acceptable for cost control, but
every included payload/cue-context block must contain its complete 2 x 2, and
the corpus totals must balance language, surface, purpose, cue family, cue
polarity, and context. Purpose-frame families and cue contexts must not be
perfectly correlated.

The generation schema and emitted metadata must record at least:

- `payload_block_id` and `matched_group_id`;
- `language` and bilingual correspondence ID;
- `intended_purpose`;
- `purpose_frame_family` and `purpose_variant_id`;
- `surface`;
- `cue_family` and cue-ontology category;
- `cue_term_pair_id` and `cue_polarity`;
- `cue_context_id` and `cue_position`;
- `alignment` (`aligned` or `crossed`);
- generator, verifier, source, and rendered-text hashes; and
- split/group identifiers used by fitting and bootstrap procedures.

The current schema has not yet been changed to implement these fields. This
section is the recorded design requirement for that later implementation.

## 7. Development and validation structure

The next protocol must use three evidence tiers:

### 7.1 Spent development material

All three confirmation banks, the five cue contexts, and the existing fitting
corpus may be used for exploratory development. They can select an estimand,
relative-depth grid, estimator family, regularization rule, and term-aware gate.
They can never provide another confirmation verdict.

### 7.2 New independent factorial confirmation

Only after every choice is frozen may another bank be authored. It must be
disjoint in payload blocks, purpose wording, cue terms, and cue-line context.
It must test both 8B stand-ins and score once with no layer, method, threshold,
or term exclusions selected from it.

The gate must include:

- purpose ordering by language, cue family, surface, and cue polarity;
- within-term purpose ordering;
- grouped lower bounds using whole payload blocks;
- literal-label and format-only invariance;
- term-level construct-adjacent cue effects;
- a prespecified hierarchical or partial-pooling family effect; and
- an anti-cancellation rule so opposing large term effects cannot produce an
  apparently small family mean.

Exact thresholds, multiplicity handling, and the relative-depth selection rule
must be declared after development on spent data and before authoring the bank.

### 7.3 Naturalistic transfer

The existing naturalistic set remains sealed. It may be opened once only after
a future controlled candidate passes its independent confirmation. If the new
estimand changes the intended construct materially, compatibility of the
existing naturalistic set must be decided without inspecting its activations or
outcomes; otherwise a new naturalistic asset must be authored prospectively.

## 8. Status and authority rules

The files under `validation/` for the three confirmation banks deliberately
retain `status: frozen_unscored`. That value records the state at which each
hash-bound authoring specification was frozen. Editing it after scoring would
invalidate the protocol artifact and its recorded hashes.

Live status is determined in this order:

1. machine-readable result under `probe_check/metrics/`;
2. the corresponding `RESULT_*.md` interpretation;
3. this archive and the claim ledger for combined current state; and
4. historical handoffs and frozen specifications for what was declared before
   execution.

Accordingly, `frozen_unscored` inside a hash-bound source specification does not
mean the bank is currently unscored. All three are spent. Historical documents
may describe a then-pending next action; later dated ledger entries and this
archive supersede those instructions without rewriting history.

## 9. Numerical authorities

| artifact | SHA-256 | role |
|---|---|---|
| `probe_check/metrics/explicit_contract_forecast.json` | `61a2f15a8509594671a4954727f7708328e8e6da309904e4b466f46befad2f29` | full explicit-only development forecast |
| `probe_check/metrics/format_matched_confirmatory_controls.json` | `e9d21a80b15fd6c85f598deca092927faf31460ca19c00b6cfa28598d505ded1` | first prospective verdict |
| `probe_check/metrics/cue_invariant_development.json` | `f8ae340fdf900748dd0ea254d7bdcb4d0b4922ba6a76cb004f9c42828919677d` | SVD90 development selection |
| `probe_check/metrics/cue_invariant_fresh_confirmation.json` | `49d2453443e26bb49260ced6f41f44edebd0f043a4631de666336414e66353c2` | second prospective verdict |
| `probe_check/metrics/cue_constrained_development.json` | `8d76b7f746f148d878e346cfbb58a48e3dbd2f1444249b5bdd7ac9745559af73` | five-context constrained development |
| `probe_check/metrics/cue_constrained_fresh_confirmation.json` | `61fd25f7e237bfbf5fcf8a3f9df880a11ee946c797ed433f516b707e0f427f3d` | third prospective verdict |

The corresponding narrative authorities are:

- `probe_check/development/layer_selection/RESULT_explicit_contract_8b.md`;
- `probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_development_8b.md`;
- `probe_check/confirmations/02_cue_invariant/RESULT_cue_invariant_confirmation_8b.md`;
- `probe_check/development/cue_constraints/RESULT_cue_constrained_development_8b.md`; and
- `probe_check/confirmations/03_cue_constrained/RESULT_cue_constrained_confirmation_8b.md`.

## 10. Next permitted work

1. Use only spent data to formalize and compare the revised term-aware estimand,
   cue ontology, relative-depth grid, and anti-cancellation gate.
2. Decide whether the project accepts the narrower controlled-separation claim
   or funds the crossed fitting-corpus redesign for the stronger claim.
3. If retaining the stronger claim, implement the schema and renderer changes,
   validate complete factorial balance, and generate a disposable development
   corpus before any new confirmation asset.
4. Obtain the outstanding native Japanese review for catalog 2.3.2 and the
   added cue-line contexts independently of the probe redesign.
5. Freeze one complete protocol. Only then author a new confirmation bank and
   request another multi-layer 8B activation extraction.

Do not tune on the third bank, inspect the naturalistic set, select a different
layer as a post-hoc rescue, or describe any rejected recipe as adopted.

## 11. Archival work record

This archive was created to replace a conversation-only synthesis with a
repository-owned decision record. The active documentation index, session
handoff, claim ledger, methods status, probe-check index, and repository README
were updated to point here and to name the third confirmation as the binding
current result. The postprocessor's emitted evaluation contract and its
regression test were also updated so a newly assembled dataset cannot describe
the rejected first-bank candidate as pending. Immutable control specifications
were not edited.

## 12. Probe-check layout migration

The probe workspace was reorganized after archival so its filesystem mirrors
the experimental chronology. Each prospective confirmation now lives under
`probe_check/confirmations/01_format_matched`, `02_cue_invariant`, or
`03_cue_constrained`, with its build script, extractor, Colab notebook,
evaluator, handoff, and result kept together. Exploratory work is separated
under `probe_check/development/baseline`, `layer_selection`, and
`cue_constraints`. Shared estimators, stable control JSONL files, compact
metrics, and local outputs remain at the probe-check root.

All executable imports, test paths, notebook workspace paths, handoff commands,
script pins, documentation links, and current references were migrated. The
four extraction preflights and the pre-generation integrity check passed after
the move. `probe_check/out/` is now ignored by Git: all 269 local activation and
log files were preserved on disk, while the compact metrics and narrative
reports remain versioned. Hash-bound historical specifications and provenance
inside completed output packages were deliberately not rewritten; they describe
the exact paths and hashes used when those artifacts were produced.
