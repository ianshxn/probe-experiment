#!/usr/bin/env python
"""Cross-stratum transfer: does a purpose signal exist that is not scrutiny?

Why this test
-------------
The implicit frames confound purpose with scrutiny: every evaluation member
carries a scrutiny-flavoured agent and verb, every deployment member a
use-flavoured one. `RESULT_controls_8b.md` shows the recovered direction follows
that vocabulary.

The EXPLICIT stratum does not have this property. Its pairs hold the verb
constant and vary purpose lexically -- "The evaluation office will use this
response" / "The deployment office will use this response". So a direction fitted
on explicit items alone is never trained on a scrutiny contrast at all.

That makes two questions answerable from activations already on disk, before
committing to any crossed redesign:

  1. Does an explicit-fitted direction TRANSFER to implicit items? If yes, a
     purpose representation is shared across strata and is not riding on the
     scrutiny vocabulary. If it transfers at chance, the only purpose signal in
     the corpus is the explicit lexeme itself, and a crossed design has little
     to isolate.

  2. Does an explicit-fitted direction INVERT on the implicit_cue controls? It
     has no reason to have learned those six verbs. If it does not invert, that
     is the sharpest available evidence that scrutiny and purpose are separable.

Both outcomes are informative. Neither is gate evidence: development corpus,
stand-in models.

Read alongside the caveat that the explicit stratum is a LEXICAL UPPER BOUND by
design -- an explicit-fitted direction is expected to be partly an
evaluation/deployment lexeme detector. Transfer to implicit items is exactly what
distinguishes "partly lexical" from "purely lexical".

Usage
-----
    uv run --no-project --with torch --with scikit-learn --with numpy `
      python probe_check/development/layer_selection/xstratum.py --acts probe_check/out/<model>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROBE_CHECK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBE_CHECK))

from analyze import (
    apply_stats,
    auc,
    bacc,
    fold_scores,
    lang_stats,
    load,
    load_acts,
)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def direction_on(X, d, mask, C):
    """Refit on every item of one stratum. Geometry only -- no folds."""
    st = lang_stats(X[mask], d["lang"][mask])
    Xs = apply_stats(X[mask], d["lang"][mask], st)
    clf = LogisticRegression(max_iter=5000, C=C).fit(Xs, d["y"][mask])
    return unit(clf.coef_[0]), clf, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--position", default="last", choices=["last", "mean", "frame"])
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg, keep, d = load(args.acts, "all")
    layers = args.layers or cfg["layers_stored"]
    expl = d["expl"]
    is_ex, is_im = expl == "explicit", expl == "implicit"
    print(f"model      : {cfg['model']}")
    print(f"item set   : {cfg.get('item_set', '?')}")
    print(f"position   : {args.position}   C={args.C:g}")
    print(f"items      : {is_ex.sum()} explicit, {is_im.sum()} implicit")

    # ------------------------------------------------------------------ 1
    print("\n=== 1. Cross-stratum transfer, leave-one-frame-family-out ===")
    print("   Frame families span both strata, so holding out a family removes")
    print("   its explicit AND implicit items from fitting.\n")
    hdr = (f"   {'layer':>5} | {'fit EXPLICIT -> implicit':>26} | "
           f"{'fit IMPLICIT -> implicit':>26} | {'fit EXPL -> explicit':>22}")
    print(hdr)
    print(f"   {'':>5} | {'en':>8}{'ja':>8}{'AUCen':>10} | "
          f"{'en':>8}{'ja':>8}{'AUCen':>10} | {'en':>8}{'ja':>8}")
    rows = {}
    for L in layers:
        X = load_acts(args.acts, L, args.position, keep)
        out = {}
        for name, fitmask, evalmask in (
            ("ex2im", is_ex, is_im),
            ("im2im", is_im, is_im),
            ("ex2ex", is_ex, is_ex),
        ):
            pred, score, _ = fold_scores(X, d, args.C, fit_mask=fitmask)
            cell = {}
            for lg in ("en", "ja"):
                m = evalmask & (d["lang"] == lg) & (pred >= 0)
                if name == "ex2ex":
                    # scored out of fold, so exclude items used to fit their own fold
                    pass
                cell[lg] = (bacc(d["y"][m], pred[m]), auc(d["y"][m], score[m]))
            out[name] = cell
        rows[L] = out
        print(f"   {L:>5} | "
              f"{out['ex2im']['en'][0]:>8.3f}{out['ex2im']['ja'][0]:>8.3f}"
              f"{out['ex2im']['en'][1]:>10.3f} | "
              f"{out['im2im']['en'][0]:>8.3f}{out['im2im']['ja'][0]:>8.3f}"
              f"{out['im2im']['en'][1]:>10.3f} | "
              f"{out['ex2ex']['en'][0]:>8.3f}{out['ex2ex']['ja'][0]:>8.3f}")

    # ------------------------------------------------------------------ 2
    print("\n=== 2. Direction geometry ===")
    print("   cosine between the explicit-fitted and implicit-fitted directions.")
    print("   Near 0 = different objects; high = one shared direction.\n")
    print(f"   {'layer':>5}{'cos(expl, impl)':>18}")
    cosines = {}
    for L in layers:
        X = load_acts(args.acts, L, args.position, keep)
        w_ex, _, _ = direction_on(X, d, is_ex, args.C)
        w_im, _, _ = direction_on(X, d, is_im, args.C)
        c = float(np.dot(w_ex, w_im))
        cosines[L] = c
        print(f"   {L:>5}{c:>18.3f}")

    # ------------------------------------------------------------------ 3
    cdir = args.acts / "controls"
    if not (cdir / "meta.jsonl").exists():
        print(f"\n(no controls at {cdir}; skipping section 3)")
        _dump(args, cfg, rows, cosines, None)
        return

    print("\n=== 3. Controls scored by an EXPLICIT-ONLY direction ===")
    print("   Threshold-free ordering AUC: do evaluation-intended crossed items")
    print("   score above deployment-intended ones? 0.5 is chance; below 0.5 means")
    print("   the direction follows the cue word against the intended purpose.")
    print("   The explicit-fitted direction never saw a scrutiny contrast.\n")

    cmeta = [json.loads(l) for l in
             (cdir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    clang = np.array([m["language"] for m in cmeta])
    cfam = np.array([m["cue_family"] for m in cmeta])
    ctype = np.array([m["control_type"] for m in cmeta])
    cpurp = np.array([m["intended_purpose"] for m in cmeta])

    print(f"   {'layer':>5} | {'implicit_cue en':>16}{'ja':>8} | "
          f"{'explicit_lexeme en':>19}{'ja':>8}")
    ctl = {}
    for L in layers:
        X = load_acts(args.acts, L, args.position, keep)
        Xc = load_acts(cdir, L, args.position)
        _, clf, st = direction_on(X, d, is_ex, args.C)
        s = clf.decision_function(apply_stats(Xc, clang, st))
        cell = {}
        for fam in ("implicit_cue", "explicit_lexeme"):
            for lg in ("en", "ja"):
                m = (cfam == fam) & (clang == lg) & (ctype == "crossed_distractor")
                lab = (cpurp[m] == "evaluation").astype(int)
                cell[(fam, lg)] = auc(lab, s[m]) if len(set(lab)) == 2 else float("nan")
        ctl[L] = cell
        print(f"   {L:>5} | {cell[('implicit_cue','en')]:>16.3f}"
              f"{cell[('implicit_cue','ja')]:>8.3f} | "
              f"{cell[('explicit_lexeme','en')]:>19.3f}"
              f"{cell[('explicit_lexeme','ja')]:>8.3f}")

    shallow = [L for L in layers if L <= 16]
    inv = sum(1 for L in shallow for lg in ("en", "ja")
              if ctl[L][("implicit_cue", lg)] < 0.5)
    print(f"\n   implicit_cue below chance in {inv}/{2*len(shallow)} cells "
          f"at layers <= 16 (explicit-fitted direction)")

    _dump(args, cfg, rows, cosines, ctl)


def _dump(args, cfg, rows, cosines, ctl):
    out = {
        "model": cfg["model"],
        "item_set": cfg.get("item_set"),
        "position": args.position,
        "C": args.C,
        "transfer": {str(L): {k: {lg: {"bacc": v[0], "auc": v[1]}
                                  for lg, v in cell.items()}
                              for k, cell in r.items()} for L, r in rows.items()},
        "cosine_explicit_implicit": {str(L): c for L, c in cosines.items()},
        "controls_explicit_fitted": (
            None if ctl is None else
            {str(L): {f"{f}|{lg}": v for (f, lg), v in cell.items()}
             for L, cell in ctl.items()}),
    }
    dest = Path("probe_check/metrics") / f"{args.acts.name}_xstratum.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
