"""Live progress view for a running ``contrastive-data generate`` pair.

The generation runner is silent until it returns, and it lives inside the
pre-generation integrity boundary, so it must not be edited to add reporting:
any change to a locked implementation module moves ``request_projection_sha256``
and resets semantic approval to pending. This watcher is a separate, unlocked
process that reads only artifacts the runner already writes.

Point it at **run A** of the pair -- all bookkeeping lands there regardless of
which model authored a given item.

    uv run python scripts/watch_generation.py runs/<run_id_a>

What it can and cannot see
--------------------------
The runner persists one atomic state file per job, but only once that job has
produced a durable outcome. A job on its first attempt has no state file yet,
so an unstarted job and an in-flight job are indistinguishable on disk. This
view reports that honestly as "waiting", bounded above by ``max_workers`` from
the runner config: at most that many of the waiting jobs are actually in
flight. Terminal statuses are ``accepted`` and ``exhausted``; everything else
is a durable mid-job checkpoint, most often ``verification_pending``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]

TERMINAL_STATUSES = ("accepted", "exhausted")
PROGRESS_STATUSES = (
    "accepted",
    "exhausted",
    "verification_pending",
    "pending",
    "malformed_advance",
    "qc_rejected_advance",
    "verification_rejected_advance",
)
BAR_WIDTH = 42


def enable_vt() -> bool:
    """Turn on ANSI handling in the Windows console; report whether redraw works."""
    if os.name == "nt":
        os.system("")
    return sys.stdout.isatty()


def load_manifest(run_dir: Path, required: bool = True) -> dict[str, dict[str, str]]:
    """Map job_id to its planned facets. The manifest is written by preflight."""
    path = run_dir / "manifest.jsonl"
    if not path.is_file():
        if not required:
            return {}
        raise SystemExit(
            f"No manifest at {path}. Point this at a preflighted run directory "
            "(run A of the pair)."
        )
    jobs: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            jobs[str(row["job_id"])] = {
                "language": str(row.get("language", "?")),
                "topic_id": str(row.get("topic_id", "?")),
                "pair_id": str(row.get("pair_id", "?")),
                "content_spec_id": str(row.get("content_spec_id", "?")),
            }
    if required and not jobs:
        raise SystemExit(f"Manifest {path} contains no jobs.")
    return jobs


def sibling_run_dir(run_dir: Path) -> Path | None:
    """Find run B of the pair, so every twin job_id can be resolved.

    Each run's manifest describes the whole grid under its own provider-derived
    job_ids, and the two id spaces are disjoint. The runner writes one state
    file per executed twin identity into run A, so roughly half of those ids
    are only found in run B's manifest. Without it the language breakdown
    silently halves.
    """
    path = CONTRASTIVE_ROOT / "configs" / "generation_run.yaml"
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        candidates = [
            (CONTRASTIVE_ROOT / str(config[key])).resolve()
            for key in ("run_dir_a", "run_dir_b")
        ]
    except Exception:
        return None
    for candidate in candidates:
        if candidate != run_dir.resolve() and candidate.is_dir():
            return candidate
    return None


LANGUAGE_FROM_PATH = re.compile(r"_(en|ja)_s\d+\.json$")


def language_of(job_id: str, facets: dict[str, dict[str, str]], state: dict) -> str:
    """Resolve a job's language from the manifests, else from its output path.

    The path fallback only covers jobs that produced content: an exhausted job
    has ``content_output_path: null`` and its attempt paths are job-id keyed,
    so its language is unrecoverable without the owning run's manifest. Such
    jobs are reported as unresolved rather than dropped, so the language rows
    always reconcile with the total.
    """
    row = facets.get(job_id)
    if row is not None:
        return row["language"]
    match = LANGUAGE_FROM_PATH.search(str(state.get("content_output_path") or ""))
    return match.group(1) if match else "?"


def read_states(
    run_dir: Path,
    facets: dict[str, dict[str, str]],
) -> list[tuple[str, str, str]]:
    """Read every persisted job state as (job_id, status, language).

    States are written atomically, so a torn read should not occur; a file can
    still vanish between listing and opening during a rename, and that is not
    an error worth interrupting the view for.
    """
    state_dir = run_dir / "state"
    if not state_dir.is_dir():
        return []
    rows: list[tuple[str, str, str]] = []
    for path in sorted(state_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        job_id = str(row.get("job_id") or path.stem)
        rows.append(
            (
                job_id,
                str(row.get("status", "pending")),
                language_of(job_id, facets, row),
            )
        )
    return rows


def parse_time(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def max_workers() -> int | None:
    """Best-effort read of the runner's concurrency, to bound in-flight jobs."""
    path = CONTRASTIVE_ROOT / "configs" / "generation_run.yaml"
    try:
        import yaml

        return int(yaml.safe_load(path.read_text(encoding="utf-8"))["max_workers"])
    except Exception:
        return None


def bar(done: int, total: int) -> str:
    filled = 0 if total == 0 else round(BAR_WIDTH * done / total)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def humanize(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render(
    run_dir: Path,
    grid: dict[str, dict[str, str]],
    states: list[tuple[str, str, str]],
    started: float,
    workers: int | None,
    finished: dict[str, object] | None,
) -> list[str]:
    # The grid is one run's manifest: it sizes the work. States are keyed by
    # twin identity and span both runs' id spaces, so they are counted, never
    # intersected with the grid.
    total = len(grid)
    counts = Counter(status for _, status, _ in states)
    settled = sum(counts[name] for name in TERMINAL_STATUSES)
    waiting = max(total - len(states), 0)

    lines = [f"{run_dir.name}  {settled}/{total} settled"]
    percent = 0.0 if total == 0 else 100.0 * settled / total
    lines.append(f"  [{bar(settled, total)}] {percent:5.1f}%")

    parts = [
        f"{name}={counts[name]}" for name in PROGRESS_STATUSES if counts[name]
    ]
    waiting_label = f"waiting={waiting}"
    if waiting and workers:
        waiting_label += f" (<={min(workers, waiting)} in flight)"
    parts.append(waiting_label)
    lines.append("  " + "  ".join(parts))

    planned = Counter(row["language"] for row in grid.values())
    for language in sorted(planned):
        seen = Counter(
            status for _, status, lang in states if lang == language
        )
        done = sum(seen[name] for name in TERMINAL_STATUSES)
        accepted = seen["accepted"]
        rate = "" if done == 0 else f"  accept={100.0 * accepted / done:5.1f}%"
        lines.append(
            f"  {language:<3} {done:>4}/{planned[language]:<4} settled   "
            f"accepted={accepted:<4} exhausted={seen['exhausted']:<4}{rate}"
        )
    unresolved = sum(1 for _, _, lang in states if lang == "?")
    if unresolved:
        lines.append(
            f"  ?   {unresolved:>4} state(s) not attributable to a language; "
            "pass --run-dir-b"
        )

    elapsed = time.time() - started
    tail = f"  elapsed {humanize(elapsed)}"
    if settled and settled < total and elapsed > 0:
        remaining = (total - settled) * (elapsed / settled)
        tail += f"   eta ~{humanize(remaining)}"
    lines.append(tail)

    if finished is not None:
        status = str(finished.get("status", "?"))
        counted = finished.get("counts", {})
        lines.append(f"  RUN {status.upper()}  {json.dumps(counted, sort_keys=True)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="run A of the generation pair")
    parser.add_argument(
        "--run-dir-b",
        help="run B, to resolve twin job ids; inferred from "
        "configs/generation_run.yaml when omitted",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one frame and exit, for logs and CI",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="append frames instead of redrawing in place",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (CONTRASTIVE_ROOT / run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"No such run directory: {run_dir}")

    grid = load_manifest(run_dir)
    if args.run_dir_b:
        other = Path(args.run_dir_b)
        if not other.is_absolute():
            other = (CONTRASTIVE_ROOT / other).resolve()
    else:
        other = sibling_run_dir(run_dir)
    facets = dict(grid)
    paired = False
    if other is not None and other.is_dir():
        sibling = load_manifest(other, required=False)
        facets.update(sibling)
        paired = bool(sibling)
    if not paired:
        print(
            "note: run B's manifest was not found, so jobs authored by the "
            "other model may not resolve to a language. Pass --run-dir-b.",
            file=sys.stderr,
        )
    workers = max_workers()
    redraw = enable_vt() and not args.plain and not args.once
    result_path = run_dir / "generation_result.json"
    started = time.time()
    previous = 0

    try:
        while True:
            states = read_states(run_dir, facets)
            finished = None
            if result_path.is_file():
                try:
                    finished = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    finished = None
            lines = render(run_dir, grid, states, started, workers, finished)

            if redraw and previous:
                sys.stdout.write(f"\x1b[{previous}A")
            for line in lines:
                sys.stdout.write(("\x1b[2K" if redraw else "") + line + "\n")
            sys.stdout.flush()
            previous = len(lines)

            if args.once or finished is not None:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
