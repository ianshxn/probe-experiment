# Held-out comprehension check

Produces the `predictions.jsonl` and `heldout_models.json` that
`contrastive-data comprehension-check` turns into the passed calibration report
required by a production preflight. This is the last armed gate: `semantic_approval`
and `frame_only` are waived in `configs/pre_generation.yaml`, `heldout_comprehension`
is not.

The gate asks one question: can an independent reader, shown one complete prompt
with no key and no paired twin, tell whether it came from the eval or the deploy
condition? If it cannot, a null probe result would be uninterpretable — we could
not separate "the subject does not represent purpose distinctly" from "the
stimulus never carried the manipulation."

## Readers

| model_id | OpenRouter slug | lineage |
|---|---|---|
| `claude_sonnet_5` | `anthropic/claude-sonnet-5` | `anthropic_claude` |
| `gpt_5_4` | `openai/gpt-5.4` | `openai_gpt` |

Both are disjoint from the generators (Qwen 3.5-397B, Gemini 3.1 Pro) and from
the three probe subjects (Llama 3.3 70B, Llama-3.3-Swallow-70B, Stockmark-2-100B),
and from each other. Change readers in `config.json`; the loader rejects repeated
lineages, and the gate itself requires at least two distinct ones.

## Run it

```powershell
$env:OPENROUTER_API_KEY = "<key>"

# Price and inspect the exact requests. No model calls.
python comprehension_check/classify.py --dataset data/processed/demo16 --dry-run

# Two items per language against both readers, to prove the wiring.
python comprehension_check/classify.py --dataset data/processed/demo16 --limit 4

# The real thing: 400 items x 2 readers = 800 requests, about $1-3 all in.
python comprehension_check/classify.py --dataset data/processed/demo16

# Gate the predictions.
uv run contrastive-data comprehension-check `
  data/processed/demo16 `
  comprehension_check/out/demo16/predictions.jsonl `
  --models comprehension_check/out/demo16/heldout_models.json `
  --output calibration/heldout_comprehension_passed.json
```

Then point `paths.heldout_comprehension_report` in `configs/pre_generation.yaml`
at the passed report.

Raw provider responses are archived per item under `out/<dataset>/raw/`, and a
rerun skips anything already archived — an interrupted run resumes rather than
repaying. `--emit-only` rebuilds the artifacts from those records without calling
anything. `out/` is gitignored.

## Before you spend anything

**The attestations are unsigned.** `config.json` ships with both
`disjoint_from_*_attestation` fields set to `false`, and the gate will reject the
evidence until they are `true`. That is deliberate: the attestation is the study
owner's claim about model provenance, not something a script should assert. Flip
them in `config.json`, then re-run with `--emit-only`.

**Choose the readers before you look at results.** The gate requires *every*
declared model to pass. Running five and reporting the two that passed is
selection on the manipulation check. If you want empirical grounds for the
choice, pilot candidates against `data/processed/demo15` or `slice220` — earlier
packets, so the choice stays blind to the demo16 items that will carry the
evidence.

**Do not touch locked files.** `pyproject.toml`, `uv.lock`, and every module
under `src/contrastive_generation/` are inside the integrity boundary. Editing
any of them moves `request_projection_sha256`, and the gate compares that hash
against the approval record — the evidence would be invalidated the moment it
was produced. That is why this directory is standard library only and lives
outside the package. Same reason the `generation:` block of
`configs/pre_generation.yaml` must not change: it is hashed into
`generation_settings_sha256`, which the gate also re-checks.

## Design choices that are load-bearing

The gate reports EN and JA separately and both must pass, so anything that could
advantage one language over the other is a confound in the study's headline
contrast.

- **The key is never read.** `classify.py` opens exactly one dataset artifact,
  `heldout_comprehension_items.jsonl`, and verifies it against the manifest hash
  before making any call.
- **One item per request.** Batching would leak the class base rate and make the
  task easier than the one the probe subject faces.
- **Monolingual instructions.** Japanese items are classified under Japanese
  instructions. English instructions wrapped around Japanese text would make
  instruction language a confound in the EN/JA comparison.
- **Counterbalanced option order.** Which class definition is listed first is
  decided per item by hashing the blind ID, so a positional response bias cannot
  show up as the class asymmetry the gate measures (it allows at most a 0.10
  eval/deploy accuracy gap).
- **Instructions name the classes with the explicit lexemes only** — "evaluation"
  and 評価, "deployment" and 運用. They deliberately avoid the implicit contrast
  verbs (精査・照合・参照 versus 処理・受け取り・利用 and their English
  counterparts). Naming those would hand the reader the answer key for the
  implicit stratum, inflating implicit accuracy and breaking the
  explicit-beats-implicit ordering gate for the wrong reason. A test enforces
  this.
- **Routing changes are refused.** If OpenRouter served more than one model
  under a reader's slug during a run, artifact emission fails rather than
  quietly attributing the evidence to one revision. `model_revision` resolves
  from what actually served, not from what was requested. Upstream *provider*
  variation under a fixed model is recorded and warned about rather than
  refused — the smoke test was routed to Claude via Amazon Bedrock, not
  Anthropic direct. Pin it if the evidence should name one serving route:
  `"provider": {"order": ["Anthropic"], "allow_fallbacks": false}` in that
  model's `request` block.
- **Partial runs are written as `predictions.partial.jsonl`**, so a smoke test
  can never be mistaken for gate evidence.

## What the gate then checks

Per reader, per language, on the implicit stratum: balanced accuracy ≥ 0.60, a
Wilson 95% lower bound above 0.50, an eval/deploy accuracy gap ≤ 0.10, and
explicit balanced accuracy strictly greater than implicit. All four, for both
readers, in both languages.

Note the consequence of that last one: a reader that is perfect everywhere
*fails*, because explicit cannot beat implicit at a ceiling. There is a test for
that.

## Tests

```powershell
uv run python -m unittest tests.test_comprehension_check -v
```

Offline, no API calls. Covers prompt construction, parsing, config validation,
packet integrity, and an end-to-end check that synthetic artifacts are accepted
by the real `validate_heldout_comprehension_results`.
