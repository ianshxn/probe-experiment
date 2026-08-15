import json, sys, numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

items, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(items) if l.strip()]
a = np.load(npz); A = a["X"]
assert [str(x) for x in a["item_ids"]] == [r["item_id"] for r in rows]
y   = np.array([int(r["label"]) for r in rows])
fam = np.array([r["purpose_family_id"] for r in rows])
fmt = np.array([r["format"] for r in rows])
blk = np.array([r["payload_block_id"] for r in rows])
pur = np.array([r["intended_purpose"] for r in rows])
ALL = np.ones(len(rows), bool)
CONF = ((pur=="evaluation")&(fmt=="benchmark")) | ((pur=="deployment")&(fmt=="casual"))

def oof(X, yy, tr, te):
    s = np.full(len(rows), np.nan); p = np.full(len(rows), -1)
    for f in sorted(set(fam)):
        trm, tem = (fam!=f)&tr, (fam==f)&te
        if not tem.any(): continue
        sc = StandardScaler()
        m = LogisticRegression(C=0.1, max_iter=3000, random_state=0)
        m.fit(sc.fit_transform(X[trm]), yy[trm])
        s[tem] = m.decision_function(sc.transform(X[tem])); p[tem] = (s[tem]>=0).astype(int)
    return s, p

def paired(s):
    d = {}
    for i, r in enumerate(rows):
        if not np.isnan(s[i]): d.setdefault((blk[i], fmt[i]), {})[r["intended_purpose"]] = s[i]
    v = [x["evaluation"]-x["deployment"] for x in d.values() if len(x)==2]
    return (float(np.mean(np.array(v)>0)), float(np.mean(v))) if v else (float("nan"), float("nan"))

def cellrates(p, m):
    o = {}
    for pu in ("evaluation","deployment"):
        for ft in ("benchmark","casual"):
            c = m & (pur==pu) & (fmt==ft)
            if c.sum(): o[f"{pu}::{ft}"] = float(p[c].mean()*100)
    return o

def one(L, X):
    X = X.astype(float); r = {"layer": int(L)}
    for name, tr, te in (("confounded", CONF, ALL), ("decorrelated", ALL, ALL),
                         ("b2c", fmt=="benchmark", fmt=="casual"),
                         ("c2b", fmt=="casual", fmt=="benchmark")):
        s, p = oof(X, y, tr, te); m = ~np.isnan(s)
        pp, pm = paired(s)
        r[name] = {"auc": float(roc_auc_score(y[m], s[m])),
                   "bal_acc": float(balanced_accuracy_score(y[m], p[m])),
                   "paired_pos": pp, "paired_mean": pm, "cells": cellrates(p, m)}
    fy = (fmt=="benchmark").astype(int)
    s, p = oof(X, fy, ALL, ALL); m = ~np.isnan(s)
    r["format"] = {"auc": float(roc_auc_score(fy[m], s[m])),
                   "bal_acc": float(balanced_accuracy_score(fy[m], p[m]))}
    print(f"L{L:02d} conf {r['confounded']['auc']:.3f} decorr {r['decorrelated']['auc']:.3f} "
          f"b2c {r['b2c']['auc']:.3f} c2b {r['c2b']['auc']:.3f} fmt {r['format']['auc']:.3f}", flush=True)
    return r

res = Parallel(n_jobs=10)(delayed(one)(L, A[:, L, :]) for L in range(A.shape[1]))
res.sort(key=lambda r: r["layer"])
json.dump(res, open(out, "w"), indent=1)
print("BEST decorr layer:", max(res, key=lambda r: r["decorrelated"]["auc"])["layer"])
print("SWEEP DONE")
