#!/usr/bin/env python3
"""Historical quick diagnostic for cue-locus probe difficulty.

This does not implement the current production contract: it uses fixed C,
last-token activations, and one caller-selected stratum. Use ``analyze.py`` for
general diagnostics and ``explicit_contract.py`` for the complete explicit-only
development forecast.

The diagnostic retains these protocol properties:

  * one direction per model, fitted across both languages, not per language;
  * five leave-one-frame-family-out outer folds;
  * per-language standardisation fitted on outer-training data only;
  * grouping respected -- a shared payload (job_id) never straddles the split.

The question is NOT "does the probe work". It is whether the *ordering* of
difficulty the held-out reader gate reports (single-locus English much harder
than both-locus) shows up in linear decodability at all.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def balanced_accuracy(y, p):
    out = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            out.append((p[m] == c).mean())
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--stratum", required=True,
                    choices=["implicit", "explicit", "all"],
                    help="required because the production fitting recipe has no default")
    args = ap.parse_args()

    meta = [json.loads(l) for l in
            (args.acts / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    cfg = json.loads((args.acts / "config.json").read_text(encoding="utf-8"))
    layers = [args.layer] if args.layer else cfg["layers_stored"]

    keep = [i for i, m in enumerate(meta)
            if args.stratum == "all" or m["explicitness"] == args.stratum]
    meta_k = [meta[i] for i in keep]
    y = np.array([1 if m["class"] == "eval" else 0 for m in meta_k])
    fam = np.array([m["frame_family_id"] for m in meta_k])
    lang = np.array([m["language"] for m in meta_k])
    locus = np.array([m["cue_locus"] for m in meta_k])
    pair = np.array([m["pair_id"] for m in meta_k])

    print(f"model={cfg['model']}  stratum={args.stratum}  n={len(meta_k)}")
    print(f"families={sorted(set(fam))}\n")

    for layer in layers:
        X_all = torch.load(args.acts / f"acts_layer{layer}.pt").numpy()[keep]
        pred = np.zeros(len(y), dtype=int)
        for held in sorted(set(fam)):
            tr, te = fam != held, fam == held
            Xtr, Xte = X_all[tr].copy(), X_all[te].copy()
            # Per-language standardisation, fitted on training data only.
            for lg in ("en", "ja"):
                mtr, mte = (lang[tr] == lg), (lang[te] == lg)
                if mtr.sum() < 2:
                    continue
                mu, sd = Xtr[mtr].mean(0), Xtr[mtr].std(0) + 1e-6
                Xtr[mtr] = (Xtr[mtr] - mu) / sd
                if mte.sum():
                    Xte[mte] = (Xte[mte] - mu) / sd
            clf = LogisticRegression(max_iter=3000, C=0.05)
            clf.fit(Xtr, y[tr])
            pred[te] = clf.predict(Xte)

        print(f"--- layer {layer} (held-out frame family, pooled fit)")
        print(f"    overall balanced accuracy      {balanced_accuracy(y, pred):.3f}")
        for lg in ("en", "ja"):
            m = lang == lg
            print(f"    {lg} balanced accuracy           {balanced_accuracy(y[m], pred[m]):.3f}")
        if args.stratum != "explicit":
            print(f"    {'':4}{'locus':12}{'EN':>8}{'JA':>8}{'n/lang':>9}")
            for lc in ("both", "verb_only", "noun_only"):
                cells = []
                for lg in ("en", "ja"):
                    m = (locus == lc) & (lang == lg)
                    cells.append(balanced_accuracy(y[m], pred[m]) if m.sum() else float("nan"))
                n = int(((locus == lc) & (lang == "en")).sum())
                print(f"    {'':4}{lc:12}{cells[0]:>8.3f}{cells[1]:>8.3f}{n:>9}")
            # per-pair English, the resolution the reader gate reports
            print(f"    {'':4}per-pair EN deploy-side recall:")
            for p in sorted(set(pair)):
                m = (pair == p) & (lang == "en") & (y == 0)
                if m.sum():
                    print(f"    {'':6}{p} {(pred[m] == 0).mean():.2f}", end="")
            print()
        print()


if __name__ == "__main__":
    main()
