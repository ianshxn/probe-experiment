#!/usr/bin/env python
"""Which (layer, read-out position) should the frozen probe use?

The contract requires layer and token position to be frozen before outcomes are
seen (`docs/PROBE_READINESS.md`), but states no CRITERION for choosing them. The
default -- take the argmax of decodability -- is not neutral on this instrument:

  * `RESULT_controls_8b.md`: the implicit_cue controls invert at exactly the
    shallow layers where decodability peaks, so argmax selects the most
    verb-confounded layer.
  * The `frame` position scores 0.90-0.95 and is degenerate: it sits before the
    payload, so under causal attention its effective sample size is the number
    of distinct frames, not the number of items.

This script evaluates every candidate cell against an ADMISSIBILITY filter
first, and reports the argmax only among survivors:

  A. effective n -- the position must actually see the payload;
  B. lexical controls -- implicit_cue and explicit_lexeme must order the crossed
     distractors correctly (threshold-free AUC >= 0.5), so the frozen cell is
     not one where the direction follows the cue vocabulary;
  C. cross-model agreement -- a cell that only works on one 8B stand-in is not a
     forecast worth freezing on.

Both development recipes are swept, because `RESULT_xstratum_8b.md` shows the
control profile is a property of the recipe: `implicit` (fit implicit, report
implicit) and `explicit` (fit explicit, report implicit). Neither is a current
production default; the generated contract now requires an explicit recipe
declaration.

Development corpus, stand-in models: a forecast for the protocol rule, never
gate evidence.

Usage (from `contrastive/`):
    uv run --no-project --with torch --with scikit-learn --with numpy `
      python probe_check/development/layer_selection/layer_position_sweep.py `
      --acts probe_check/out/Meta-Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xstratum import direction_on


def effective_n(X, d, mask):
    """Distinct activation vectors, and within-frame-group spread share.

    A read-out position that lies before the payload is a pure function of the
    frame text, so items sharing a frame collapse to one vector. Both numbers
    detect that: `distinct` falls to the frame count, and `within_share` (mean
    within-group variance over total variance) falls toward zero.
    """
    out = {}
    for lg in ("en", "ja"):
        m = mask & (d["lang"] == lg)
        Xm = X[m]
        distinct = len(np.unique(Xm, axis=0))
        groups = np.array([f"{p}|{y}" for p, y in zip(d["pair"][m], d["y"][m])])
        tot = Xm.var(0).sum()
        within = np.mean([Xm[groups == g].var(0).sum()
                          for g in np.unique(groups) if (groups == g).sum() > 1])
        out[lg] = {"n": int(m.sum()), "distinct": int(distinct),
                   "distinct_frac": float(distinct / max(m.sum(), 1)),
                   "within_share": float(within / tot) if tot else float("nan")}
    return out


def controls_auc(X, Xc, d, cmeta_arrays, fit_mask, C):
    """Threshold-free ordering AUC on the crossed distractors.

    Scored with a direction refitted on the whole fitting stratum, matching
    `xstratum.py` section 3. Below 0.5 means the direction follows the cue word
    against the intended purpose.
    """
    clang, cfam, ctype, cpurp = cmeta_arrays
    _, clf, st = direction_on(X, d, fit_mask, C)
    s = clf.decision_function(apply_stats(Xc, clang, st))
    cell = {}
    for fam in ("implicit_cue", "explicit_lexeme"):
        for lg in ("en", "ja"):
            m = (cfam == fam) & (clang == lg) & (ctype == "crossed_distractor")
            lab = (cpurp[m] == "evaluation").astype(int)
            cell[f"{fam}|{lg}"] = auc(lab, s[m]) if len(set(lab)) == 2 else float("nan")
    return cell


def decode(X, d, fit_mask, eval_mask, C):
    """Out-of-fold decodability on the reporting stratum, plus its worst cell."""
    pred, score, _ = fold_scores(X, d, C, fit_mask=fit_mask)
    res = {}
    for lg in ("en", "ja"):
        m = eval_mask & (d["lang"] == lg) & (pred >= 0)
        res[lg] = {"bacc": bacc(d["y"][m], pred[m]), "auc": auc(d["y"][m], score[m])}
    # worst fold x language cell -- stability, not just the pooled headline
    cells = []
    for fam in np.unique(d["fam"]):
        for lg in ("en", "ja"):
            m = eval_mask & (d["fam"] == fam) & (d["lang"] == lg) & (pred >= 0)
            if m.sum() >= 4 and len(set(d["y"][m])) == 2:
                cells.append(bacc(d["y"][m], pred[m]))
    res["worst_cell"] = float(min(cells)) if cells else float("nan")
    res["min_lang"] = float(min(res["en"]["bacc"], res["ja"]["bacc"]))
    res["lang_gap"] = float(abs(res["en"]["bacc"] - res["ja"]["bacc"]))
    # noun_only is the weak stratum the frame rewrite targets; track it here
    for locus in ("both", "verb_only", "noun_only"):
        for lg in ("en", "ja"):
            m = eval_mask & (d["locus"] == locus) & (d["lang"] == lg) & (pred >= 0)
            if m.sum() >= 4 and len(set(d["y"][m])) == 2:
                res.setdefault("locus", {})[f"{locus}|{lg}"] = bacc(d["y"][m], pred[m])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--positions", nargs="*", default=["last", "mean", "frame"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("probe_check/metrics"))
    args = ap.parse_args()

    cfg, keep, d = load(args.acts, "all")
    layers = args.layers or cfg["layers_stored"]
    is_ex, is_im = d["expl"] == "explicit", d["expl"] == "implicit"

    cdir = args.acts / "controls"
    has_ctl = (cdir / "meta.jsonl").exists()
    cmeta_arrays = None
    if has_ctl:
        cmeta = [json.loads(l) for l in
                 (cdir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
        cmeta_arrays = (
            np.array([m["language"] for m in cmeta]),
            np.array([m["cue_family"] for m in cmeta]),
            np.array([m["control_type"] for m in cmeta]),
            np.array([m["intended_purpose"] for m in cmeta]),
        )

    print(f"model    : {cfg['model']}")
    print(f"item set : {cfg.get('item_set', '?')}   C={args.C:g}")
    print(f"items    : {is_ex.sum()} explicit, {is_im.sum()} implicit"
          f"   controls: {'yes' if has_ctl else 'NO'}")

    rows = {}
    for pos in args.positions:
        print(f"\n=== position '{pos}' "
              f"{'=' * 52}")
        print(f"   {'layer':>5} | {'recipe':>8} | {'impl en':>8}{'ja':>8}"
              f"{'worst':>8} | {'impl_cue en':>12}{'ja':>7} |"
              f"{'expl_lex en':>12}{'ja':>7} | admissible")
        for L in layers:
            X = load_acts(args.acts, L, pos, keep)
            eff = effective_n(X, d, is_im)
            for recipe, fit_mask in (("implicit", is_im), ("explicit", is_ex)):
                dec = decode(X, d, fit_mask, is_im, args.C)
                ctl = (controls_auc(X, load_acts(cdir, L, pos), d,
                                    cmeta_arrays, fit_mask, args.C)
                       if has_ctl else {})
                sees_payload = min(v["distinct_frac"] for v in eff.values()) > 0.9
                ctl_ok = (all(v >= 0.5 for v in ctl.values())
                          if ctl and not any(np.isnan(list(ctl.values())))
                          else None)
                adm = sees_payload and bool(ctl_ok)
                rows[f"{pos}|{L}|{recipe}"] = {
                    "position": pos, "layer": L, "recipe": recipe,
                    "decode": dec, "controls": ctl, "effective_n": eff,
                    "sees_payload": bool(sees_payload),
                    "controls_pass": ctl_ok,
                    "admissible": bool(adm),
                }
                c = ctl or {}
                fmt = lambda k: f"{c[k]:>7.3f}" if k in c else f"{'--':>7}"
                print(f"   {L:>5} | {recipe:>8} | "
                      f"{dec['en']['bacc']:>8.3f}{dec['ja']['bacc']:>8.3f}"
                      f"{dec['worst_cell']:>8.3f} | "
                      f"{fmt('implicit_cue|en'):>12}{fmt('implicit_cue|ja')} |"
                      f"{fmt('explicit_lexeme|en'):>12}{fmt('explicit_lexeme|ja')} | "
                      f"{'YES' if adm else 'no'}"
                      f"{'' if sees_payload else '  (degenerate position)'}")
            e = eff["en"]
            if e["distinct_frac"] <= 0.9:
                print(f"   {'':>5} | {'':>8}   effective n: {e['distinct']}/{e['n']} "
                      f"distinct EN vectors, within-frame spread "
                      f"{e['within_share']:.1%} of total")

    dest = args.out / f"{args.acts.name}_layerpos.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"model": cfg["model"], "item_set": cfg.get("item_set"), "C": args.C,
         "cells": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")

    adm = [r for r in rows.values() if r["admissible"]]
    print(f"\n{len(adm)}/{len(rows)} cells admissible.")
    for r in sorted(adm, key=lambda r: -r["decode"]["min_lang"])[:8]:
        print(f"   {r['position']:>5} L{r['layer']:<3} {r['recipe']:>8}  "
              f"min-lang {r['decode']['min_lang']:.3f}  "
              f"worst {r['decode']['worst_cell']:.3f}  "
              f"gap {r['decode']['lang_gap']:.3f}")


if __name__ == "__main__":
    main()
