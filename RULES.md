# Operational Rules

These rules are mandatory for any agent working on this project.

## 1. Storage boundary: `/jppatton/langea/` only

On the `uahpc` cluster, work **only** inside the `jppatton/langea` subtree on each filesystem:

```text
/home/jppatton/langea/
/bighome/jppatton/langea/
/scratch/jppatton/langea/
```

Do not create, modify, cache, clone, stage, or delete project files anywhere else under `/home`, `/bighome`, or `/scratch`.

Storage guidance:

| Filesystem | Approximate quota | Allowed project root |
|---|---:|---|
| `/home` | 20 GB | `/home/jppatton/langea/` only |
| `/bighome` | 100 GB | `/bighome/jppatton/langea/` only |
| `/scratch` | 500 GB | `/scratch/jppatton/langea/` only |

The quota numbers are operational guidance, not permission to use other directories. The path rule is stricter than the quota rule: use only the matching `/jppatton/langea/` subtree.

Recommended placement:

- Source checkout, lightweight configuration, and persistent small metadata: `/home/jppatton/langea/`.
- Larger persistent model/cache material: `/bighome/jppatton/langea/`.
- SLURM runs, generated outputs, temporary logs, and scratch data: `/scratch/jppatton/langea/`.

Before a cluster command that writes files, verify its destination begins with one of the three allowed prefixes. Never use an unscoped temporary directory or a default cache outside these prefixes. Set cache and temporary environment variables explicitly when needed, for example:

```bash
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
```

Do not place credentials, model caches, logs, or generated artifacts directly in `/home`, `/bighome`, or `/scratch` outside the corresponding `jppatton/langea` directory.

## 2. API keys and `.env.local`

The local repository uses a root-level `.env.local` file for development credentials. It is ignored/private configuration and must never be committed, copied into a report, printed, or pasted into chat or logs.

The currently configured variable names are:

```text
GCLOUD_API_KEY
GOOGLE_CLOUD_PROJECT
HF_TOKEN
NVIDIA_API_KEY
OPENROUTER_API_KEY
```

Rules:

- Never reveal key values, even partially.
- Never run a command that prints the contents of `.env.local`.
- Never include `.env.local` in an archive, artifact, notebook output, issue, commit, or handoff.
- Never hard-code a key in Python, YAML, shell scripts, prompts, or SLURM files.
- Load the file only locally and only into the current process environment.
- Use the repository's existing provider configuration and environment-variable names; do not invent aliases casually.
- Check credential presence without printing values, for example by printing only whether a variable is set.
- If a credential is exposed, stop using it and report that rotation is required; do not copy the exposed value elsewhere.

For local commands, load `.env.local` through the existing project convention or an equivalent process-local loader. Do not create a second credential file. Cluster jobs must use the cluster's approved credential setup and must not copy local `.env.local` to the cluster.

Provider meaning in this project:

- `GCLOUD_API_KEY`: Google/Vertex route credential where the active provider configuration requires API-key authentication.
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project identifier; not itself a secret.
- `HF_TOKEN`: Hugging Face authentication for model or dataset access when required.
- `NVIDIA_API_KEY`: NVIDIA NIM route credential.
- `OPENROUTER_API_KEY`: OpenRouter route credential.

Before any paid or remote API run:

1. Inspect the immutable run configuration and provider route.
2. Confirm the expected environment variable is set without printing it.
3. Run the smallest available connectivity or preflight check.
4. Record provider, model, revision, route, and run ID in the artifact, but never record the key.
5. Stop if the route, model, or billing state is unexpected.

## 3. SSH access to `uahpc`

The configured SSH alias is `uahpc`:

```bash
ssh uahpc
```

Use the alias rather than embedding usernames, hostnames, passwords, tokens, or private-key paths in project files. A normal interactive session should immediately move to an allowed project directory:

```bash
ssh uahpc
cd /scratch/jppatton/langea/deploy
```

For a one-shot command:

```bash
ssh uahpc "cd /scratch/jppatton/langea/deploy && pwd && <command>"
```

For SLURM inspection:

```bash
ssh uahpc "squeue -u \$USER"
ssh uahpc "sacct -u \$USER --starttime today --format=JobID,State,Elapsed,ExitCode -P"
```

Use only project-owned paths in remote commands, including source, logs, caches, results, and temporary files. The project’s standard locations are:

```text
source/deployment: /scratch/jppatton/langea/deploy/
runs:             /scratch/jppatton/langea/runs/
results:          /scratch/jppatton/langea/results/
logs:             /scratch/jppatton/langea/logs/
HF cache:         /scratch/jppatton/langea/huggingface/hub/
cluster Python:   /bighome/jppatton/langea/runtime/vllm/bin/python
credentials:     /home/jppatton/langea/credentials/   # do not print or copy
```

The credentials directory is readable only as needed for approved jobs. Never display its contents or include its path's secret material in diagnostics.

When using `scp` or `rsync`, both source and destination must be explicitly checked. Remote destinations must begin with `/home/jppatton/langea/`, `/bighome/jppatton/langea/`, or `/scratch/jppatton/langea/`:

```bash
scp local_file uahpc:/scratch/jppatton/langea/incoming/
rsync -av local_dir/ uahpc:/scratch/jppatton/langea/incoming/local_dir/
```

Never use `~`, `/tmp`, `/var/tmp`, a bare `/home`, a bare `/bighome`, or a bare `/scratch` as a project destination. Expand and inspect the path first.

## 4. Safety and provenance

- Inspect `squeue`, `sacct`, logs, and run state before changing or stopping jobs.
- Do not overwrite immutable run directories; use a new run ID for retries.
- Do not reset or bulk-clean the worktree or cluster directories.
- Preserve manifests, hashes, model revisions, quantization, compute details, and job IDs.
- Do not invent missing stimulus content or silently recode unresolved outputs.
- Keep secrets out of generated reports and provider-response artifacts.
- If a command would write outside the three approved `jppatton/langea` roots, stop and correct it before execution.

# 5. TAE 2026 sprint mission

Target venue: **TAE (Trust-AI-Eval), NeurIPS 2026 workshop**. Submission deadline: **August 29, 2026 AoE**.

The sprint is not trying to resurrect the failed `jProbe` confirmatory claim. The workshop target is narrower and mechanistic:

> Evaluation-context representations contain a format-shared purpose component and a format-conditioned interaction component. Naive probes can collapse onto format; decorrelated training recovers purpose; imperfect cross-format transfer should be explained by the geometry and causal role of the shared and interaction components.

Required progression:

1. validate the phenomenon without sample-size confounds;
2. identify the factorial geometry that explains it;
3. causally intervene on the identified components;
4. replicate the model-comparison trajectory cleanly enough for a workshop paper;
5. write the paper around the strongest defensible claim, including negative/null results.

Do not optimize for a predetermined positive result. Optimize for a clean answer.

## 5.1 Repository and evidence boundaries

Primary repository for this sprint: `probe-experiment`.

`jProbe` may be used as a source of already-debugged causal-endpoint / steering implementation ideas, but:

- do not silently copy scientific results from `jProbe`;
- do not treat `jProbe` development or failed confirmatory runs as evidence for this paper;
- do not mix datasets, activations, directions, or run artifacts without explicit hashes and provenance;
- if code is ported, record the source commit and re-test it in this repository.

Existing `probe-experiment` 8B/70B results are development evidence. Never call an exploratory result confirmatory.

# 6. Scientific object: the 2 x 2 factorial design

Every canonical payload is crossed with:

- purpose: `evaluation` vs `deployment`
- format: `benchmark` vs `casual`

For a matched payload at a layer, define:

- `h_EB`: evaluation + benchmark
- `h_EC`: evaluation + casual
- `h_DB`: deployment + benchmark
- `h_DC`: deployment + casual

Use effect coding with purpose `p in {+1,-1}` and format `f in {+1,-1}`:

```text
h_(p,f) = mu + p*A + f*B + p*f*C + epsilon
```

where:

- `A`: format-shared purpose main effect
- `B`: format main effect / offset
- `C`: purpose x format interaction

Matched purpose contrasts:

```text
delta_B = h_EB - h_DB = 2(A + C)
delta_C = h_EC - h_DC = 2(A - C)
A = (delta_B + delta_C) / 4
C = (delta_B - delta_C) / 4
B = ((h_EB + h_DB) - (h_EC + h_DC)) / 4
```

This decomposition is the sprint's primary mechanistic hypothesis. Do not silently redefine it after inspecting results.

# 7. Primary scientific questions

## Q1. Does the pooled/decorrelated advantage survive an equal-N control?

The current analysis trains `decorrelated` on both formats (`ALL`) but trains B->C / C->B on one format only. Therefore the raw performance gap is confounded by both training-set size and target-format exposure.

Before interpreting the gap mechanistically, run an **equal-N mixed-format control**.

Preferred control:

- preserve the same number of training payload blocks as the one-format regime;
- for each training payload block, use the eval/deploy pair from exactly one assigned format;
- balance benchmark/casual assignments across blocks;
- repeat over a fixed, recorded seed bank;
- keep held-out purpose families completely held out;
- compare against benchmark-only and casual-only with identical training-row count.

Report **AUC and balanced accuracy separately**.

Do not claim a format-conditioned representation from the raw unequal-N gap.

## Q2. Is cross-format degradation explained by B and/or C?

Across all layers, on held-out purpose families / payload blocks, measure at minimum:

- cosine(delta_B, delta_C)
- cosine of mean benchmark/casual purpose directions
- norm(A)
- norm(B)
- norm(C)
- norm(C) / norm(A)
- principal angles / top principal cosines between benchmark and casual purpose-delta subspaces
- cross-format AUC
- cross-format balanced accuracy
- equal-N mixed-format AUC
- projection of format offset B onto each learned purpose direction

Primary explanatory tests:

1. Does higher `norm(C)/norm(A)` track lower cross-format **AUC**?
2. Does larger purpose-probe projection onto `B` track cross-format **threshold / balanced-accuracy failure**?
3. Do these relationships survive bootstrap over complete payload blocks / purpose families and fixed-dimensional projection/PCA controls?

Treat layers as repeated measurements from one model, not as independent replicates. Do not manufacture tiny p-values by treating 32 or 80 layers as N=32/N=80 independent samples.

## Q3. Does causal manipulation support the decomposition?

This is the desired workshop-level mechanistic result.

Select causal layers by a **predeclared rule** using development geometry, before looking at causal outcomes. A reasonable default rule is:

- restrict to layers with adequate held-out purpose decodability;
- choose one layer with low interaction ratio `norm(C)/norm(A)`;
- choose one layer with high interaction ratio `norm(C)/norm(A)`;
- write the exact selection rule into an immutable run config before causal inference.

Fit A, B, and C using training purpose families only. Use held-out purpose families for causal evaluation.

Preferred causal predictions:

- `+A` moves an independent evaluation/deployment readout in the **same direction under benchmark and casual formats**;
- `-A` reverses that movement;
- `+C` / `-C` produce **format-dependent or opposite-signed purpose effects**, consistent with the `p*f*C` term;
- `+B` / `-B` primarily affect a format readout / calibration offset and should not masquerade as a format-invariant purpose intervention.

If an independent direct-logit endpoint is used, validate its unsteered accuracy and calibration first. A causal claim is invalid if the endpoint itself is near chance or wrapper-fragile.

If behavioral output is not reliable enough, a secondary causal target may be propagation into later-layer purpose representations, but label it **internal causal mediation**, not behavioral steering.

Random-direction controls and sign-symmetric interventions are required for any causal claim.

# 8. Model comparison

The existing matched observational comparison is:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`

on the same rendered item bank and held-out-family analysis.

This is useful but does **not** identify scale, because model size is confounded with model generation/training.

Priority replication:

1. Llama-3.1-8B-Instruct
2. **Llama-3.1-70B-Instruct**
3. Llama-3.3-70B-Instruct

Use, as nearly as technically possible:

- exact same rendered items and order;
- exact same item hashes;
- same chat-template semantics;
- final prompt token;
- zero-based transformer-block output before final model norm;
- BF16 inference;
- float32 saved activations;
- all transformer layers;
- identical leave-one-purpose-family-out splits;
- identical probe regularization and analysis code;
- identical equal-N control;
- identical geometry analysis.

Interpretation:

- 3.1-8B vs 3.1-70B is the cleaner **within-family scale comparison**;
- 3.1-70B vs 3.3-70B is the cleaner **same-size generation/training comparison**.

Do not call either comparison perfectly causal. State exactly what is controlled.

The 3.1-70B extraction runs in parallel with the cheap offline mechanistic analyses and must not block them.

# 9. No p-hacking / no result laundering

Forbidden:

- changing C, layer, seed bank, normalization, split, endpoint wording, or success criterion because a result looks weak;
- reporting the best layer without the full trajectory;
- treating the best of many seeds as the result;
- silently dropping hard purpose families or payload blocks;
- using row-level random splits;
- interpreting AUC and threshold accuracy as interchangeable;
- describing observational geometry as causal;
- describing development evidence as confirmation;
- hiding a failed intervention.

If a planned analysis changes after seeing data, create a new clearly labeled exploratory protocol and state why.

Negative results are valid workshop results if the measurement lesson is clear.

# 10. Statistical unit and resampling

Keep complete matched payload blocks together.

Primary grouping/generalization unit is the purpose-family-held-out structure already used by the experiment.

For uncertainty:

- bootstrap complete payload blocks or complete purpose families as appropriate;
- preserve the four-cell structure within each resampled payload;
- never bootstrap individual rendered rows independently;
- use fixed seeds recorded in the run config;
- report seed bank and number of resamples.

For model-depth trajectories, prefer effect curves with bootstrap bands and predeclared summary statistics such as area-under-depth-curve or mean over a fixed relative-depth interval.

# 11. Subagent orchestration

The parent agent owns scientific integration and final decisions.

Spawn subagents immediately. Give each a non-overlapping responsibility, explicit files/run IDs, and a required machine-readable deliverable.

Recommended minimum team:

## A. Equal-N / statistics subagent

Own:

- equal-N mixed-format control;
- deterministic seed bank;
- AUC vs balanced-accuracy decomposition;
- block/family bootstrap;
- diagnostics linking geometry to transfer.

Deliver result JSON + short markdown interpretation.

## B. Factorial geometry subagent

Own:

- A/B/C decomposition;
- all-layer geometry;
- delta cosine;
- direction cosine;
- interaction ratio;
- principal-angle/subspace analysis;
- fixed-dimensional projection/PCA controls;
- publication-quality plots.

No causal claims.

## C. Causal mechanism subagent

Own:

- port or implement residual-stream intervention;
- independent endpoint validation;
- preregistered layer-selection rule;
- `+/-A`, `+/-B`, `+/-C`;
- random-direction controls;
- held-out-family evaluation;
- immutable causal run artifacts.

This subagent must not choose layers based on causal outcomes.

## D. Llama-3.1-70B compute subagent

Own:

- exact stimulus-hash verification;
- BF16 all-layer extraction;
- SLURM lifecycle;
- GPU resource adaptation;
- output hashes;
- exact same sweep and geometry pipeline.

Do not redesign the dataset.

## E. Reproducibility / hostile-review subagent

Own:

- split leakage checks;
- exact four-cell reconstruction;
- model/revision/dtype/layer-semantics verification;
- accidental unequal-N detection;
- immutable-run policy;
- regression tests;
- adversarial attempts to falsify every headline claim.

## F. TAE paper subagent

Own from day one:

- NeurIPS 2026 double-blind workshop LaTeX skeleton;
- figure/table placeholders;
- related-work matrix;
- methods text from immutable configs;
- limitations;
- anonymization checks.

Do not wait for the final experiment to start writing.

Subagents may create their own subagents when it accelerates a bounded task.

Avoid simultaneous edits to the same file. Use isolated branches/worktrees or explicit file ownership, then integrate intentionally.

# 12. Required sprint artifacts

Create a sprint root such as:

```text
runs/tae_2026/
```

with immutable run IDs below it. At minimum preserve:

```text
runs/tae_2026/
  sprint_manifest.json
  equal_n/
  geometry/
  causal/
  llama31_70b/
  model_comparison/
  hostile_review/
  figures/
  paper_tables/
```

Each scientific result artifact must record:

- git commit
- run ID
- source item SHA256
- model and exact revision
- dtype
- layer semantics
- token position
- split definition
- regularization
- seed(s)
- input artifact hashes
- output artifact hashes
- whether result is exploratory/development or confirmation-grade

Never overwrite a result artifact. Use a new run ID.

# 13. Testing and falsification

Every new analysis must have synthetic tests demonstrating that it detects the intended effect and does not create the effect when the component is absent.

Required synthetic cases for factorial decomposition:

1. `A only`: cross-format transfer strong; estimated C near zero.
2. `B only`: format decodable; purpose not decodable.
3. `C only`: format-conditioned/opposite purpose code; pooled/shared A near zero.
4. `A + B`: threshold shift possible without directional rotation.
5. `A + C`: cross-format directional mismatch predicted.
6. null/noise: no systematic purpose result.

For causal code, test:

- hook fires exactly once;
- only intended token/layer changes;
- sign reversal works;
- zero-strength reproduces baseline;
- random controls use the same direction for +/- signs;
- endpoint logits are finite and candidate mapping is correct.

The hostile-review subagent should try to explain every headline result as:

- format leakage;
- sample-size advantage;
- target-format exposure;
- payload identity leakage;
- family leakage;
- threshold miscalibration;
- dimensionality/covariance artifact;
- probe regularization artifact;
- layer-selection bias;
- endpoint bias.

Do not write the headline until these alternatives are addressed.

# 14. TAE paper constraints

Prepare an anonymized NeurIPS 2026 workshop submission.

Target:

- maximum 8 main-text pages, excluding references/appendices;
- single PDF;
- double blind;
- official NeurIPS 2026 style;
- `\usepackage[dblblindworkshop]{neurips_2026}`;
- `\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}`.

Paper structure should converge toward:

1. evaluation-awareness probes can confound purpose with benchmark format;
2. controlled factorial design;
3. confounded vs decorrelated vs cross-format phenomenon, with equal-N control;
4. factorial decomposition into shared purpose, format, and interaction components;
5. mechanistic explanation of ranking loss vs threshold shift;
6. causal intervention on shared vs interaction components;
7. model-size/training comparison;
8. implications for validity of evaluation-awareness measurement;
9. limitations and what remains unproven.

Prioritize a clean, falsifiable methodological contribution over scope creep.

# 15. Sprint stop conditions

A useful completion state is one of:

## `TAE_MECHANISTIC_RESULT_STRONG`

Equal-N phenomenon survives, factorial geometry predicts the transfer failure, and causal interventions behave as predicted with appropriate controls.

## `TAE_MECHANISTIC_RESULT_PARTIAL`

Geometry cleanly explains the phenomenon but behavioral causal intervention is weak/unstable; paper is framed as measurement + mechanistic representation analysis, with internal causal mediation only if valid.

## `TAE_NEGATIVE_RESULT_CLEAN`

Equal-N control removes the gap or causal predictions fail. Preserve this result and frame the paper around what the original probe comparison falsely suggested and why evaluation methodology needs stronger controls.

## `TAE_TECHNICALLY_BLOCKED`

Only if a real external infrastructure/access problem prevents the required experiment after reasonable technical recovery attempts.

Do not invent success.
