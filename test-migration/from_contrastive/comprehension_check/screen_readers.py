#!/usr/bin/env python3
"""Screen candidate held-out readers on the explicit stratum only.

A reader is only a usable instrument for the comprehension gate if it can read
frames that *literally name* evaluation or deployment. Claude Sonnet 5 scored
0.67 on those items and GPT-5.4 0.90-0.92, while DeepSeek V4 Pro scored
0.98-1.00 — a spread that has nothing to do with the stimulus, since the text in
question cannot be made more explicit.

This screen measures exactly that, and nothing else. It never touches the
implicit stratum the gate scores, so choosing readers on its output is not
selection on the gated outcome.

Blinding: the reader still sees only item text, rendered by the same
`render_messages` the gate uses, with the same per-item counterbalanced option
order. The answer key is read by this selector — to pick out explicit items and
to score them — but never reaches a model. `classify.py` remains provably
key-blind; that invariant belongs to the gate path and is deliberately not
loosened here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cc_classify", HERE / "classify.py")
assert _spec and _spec.loader
classify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classify)

KEY_ARTIFACT = "heldout_comprehension_key.jsonl"


def load_explicit(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return the explicit-stratum items plus their key rows."""
    items = classify.load_items(dataset_dir)          # hash-checked, key-blind
    key_path = dataset_dir / KEY_ARTIFACT
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if classify.sha256_file(key_path) != manifest["artifacts"][KEY_ARTIFACT]:
        raise classify.ClassifyError(f"{KEY_ARTIFACT} does not match its manifest hash")
    key = {str(r["blind_item_id"]): r for r in classify.read_jsonl(key_path)}
    explicit = [i for i in items if key[str(i["blind_item_id"])]["explicitness"] == "explicit"]
    if not explicit:
        raise classify.ClassifyError("Packet contains no explicit-stratum items")
    return explicit, key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=HERE / "screen_config.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args(argv)

    dataset = args.dataset.resolve()
    out_dir = (args.out or HERE / "out" / f"screen_{dataset.name}").resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["models"]
    if args.model:
        keys = {m["model_key"] for m in models}
        unknown = sorted(set(args.model) - keys)
        if unknown:
            raise classify.ClassifyError(f"Unknown --model values: {unknown}")
        models = [m for m in models if m["model_key"] in args.model]
    floor = float(config.get("eligibility_floor", 0.95))

    explicit, key = load_explicit(dataset)
    explicit = classify.select_items(explicit, args.limit)
    print(f"{len(explicit)} explicit items x {len(models)} candidates = "
          f"{len(explicit) * len(models)} requests")

    api_key = os.environ.get(classify.API_KEY_ENV)
    if not api_key:
        raise classify.ClassifyError(f"{classify.API_KEY_ENV} is not set")

    pending = []
    for model in models:
        settings = classify.resolved_request(config, model)
        fingerprint = classify.settings_fingerprint(settings)
        for item in explicit:
            path = classify.record_path(out_dir, str(model["model_key"]), str(item["blind_item_id"]))
            if path.is_file():
                archived = json.loads(path.read_text(encoding="utf-8"))
                if archived.get("settings_sha256") == fingerprint:
                    continue
            pending.append((item, model, settings))
    print(f"{len(pending)} to call ({len(explicit) * len(models) - len(pending)} archived)")

    failures: dict[str, str] = {}
    lock = threading.Lock()
    done = {"n": 0}

    def work(job) -> None:
        item, model, settings = job
        key_name = str(model["model_key"])
        try:
            record = classify.classify_one(item, model, settings, api_key)
        except classify.ClassifyError as exc:
            with lock:
                failures.setdefault(key_name, str(exc)[:160])
            return
        classify.write_atomic(
            classify.record_path(out_dir, key_name, str(item["blind_item_id"])),
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        )
        with lock:
            done["n"] += 1
            if done["n"] % 50 == 0:
                print(f"  {done['n']}/{len(pending)}", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            list(pool.map(work, pending))

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": classify.utc_now(),
        "dataset_dir": str(dataset),
        "stratum": "explicit_only",
        "eligibility_floor": floor,
        "items_screened": len(explicit),
        "candidates": {},
    }
    print(f"\n{'candidate':<16}{'lineage':<17}{'EN eval':>8}{'EN dep':>8}"
          f"{'JA eval':>8}{'JA dep':>8}{'min':>7}  verdict")
    for model in models:
        mk = str(model["model_key"])
        records = {}
        for item in explicit:
            p = classify.record_path(out_dir, mk, str(item["blind_item_id"]))
            if p.is_file():
                r = json.loads(p.read_text(encoding="utf-8"))
                records[r["blind_item_id"]] = r
        if len(records) < len(explicit):
            note = failures.get(mk, "incomplete")
            print(f"{mk:<16}{model['lineage']:<17}{'—':>8}{'—':>8}{'—':>8}{'—':>8}{'—':>7}  "
                  f"UNSCREENED ({len(records)}/{len(explicit)}) {note[:60]}")
            report["candidates"][mk] = {"status": "incomplete", "classified": len(records),
                                        "error": failures.get(mk)}
            continue
        cells: dict[tuple[str, str], list[bool]] = {}
        for bid, r in records.items():
            k = key[bid]
            cells.setdefault((k["language"], k["class"]), []).append(r["prediction"] == k["class"])
        acc = {f"{l}_{c}": sum(v) / len(v) for (l, c), v in cells.items()}
        worst = min(acc.values())
        ok = worst >= floor
        usage_in = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in records.values())
        usage_out = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records.values())
        pr = model.get("pricing_usd_per_mtok", {})
        report["candidates"][mk] = {
            "status": "qualified" if ok else "rejected",
            "lineage": model["lineage"], "openrouter_slug": model["openrouter_slug"],
            "accuracy": acc, "min_cell_accuracy": worst,
            "served_models": sorted({str(r.get("served_model")) for r in records.values()}),
            "served_providers": sorted({str(r.get("served_provider")) for r in records.values()}),
            "observed_usd": round(usage_in / 1e6 * float(pr.get("input", 0))
                                  + usage_out / 1e6 * float(pr.get("output", 0)), 4),
        }
        print(f"{mk:<16}{model['lineage']:<17}{acc.get('en_eval', 0):>8.2f}"
              f"{acc.get('en_deploy', 0):>8.2f}{acc.get('ja_eval', 0):>8.2f}"
              f"{acc.get('ja_deploy', 0):>8.2f}{worst:>7.2f}  "
              f"{'QUALIFIES' if ok else 'rejected'}")

    classify.write_atomic(out_dir / "screen_report.json",
                          json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    qualified = [k for k, v in report["candidates"].items() if v.get("status") == "qualified"]
    print(f"\nqualified: {qualified or 'none'}")
    print(f"wrote {out_dir / 'screen_report.json'}")
    print("\nA qualifying candidate is an eligible reader, not a declared one. Declaring it "
          "still requires a lineage attestation and a slot not already held by another "
          "declared reader.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except classify.ClassifyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
