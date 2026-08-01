#!/usr/bin/env python3
"""
CAPTURE-PLANNING PROTOCOL - is a tiny pilot enough to decide which PAIs to capture?

Capturing a PAI is expensive (bespoke masks, makeup sessions, printing). If the axes estimated
from FEW images per PAI already select the same subset as the full data, capture can be planned:
small pilot -> geometry -> decide what to scale up.

Test (CPU only, no training): subsample n images/PAI, re-estimate the axes, re-run the selection,
and measure the agreement with the full-data choice (Jaccard) and the real coverage lost
(always evaluated with the full geometry, which is the ground truth).

Usage: python 09_pilot.py [HQ-WMCA]
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daxis_ext import standardize
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
SIZES = [25, 50, 100, 200, 500]
REPS = 20


def axes_from(Xs, y, st, ks, rng, n_per):
    """Axes from n_per images per PAI (and the same count of bonafide): simulates a pilot."""
    live = np.where(y == 0)[0]
    lv = rng.choice(live, min(len(live), n_per * 2), replace=False)
    mu_live = Xs[lv].mean(0)
    A = {}
    for k in ks:
        idx = np.where((st == k) & (y == 1))[0]
        if len(idx) == 0: continue
        s = rng.choice(idx, min(len(idx), n_per), replace=False)
        d = Xs[s].mean(0) - mu_live
        A[k] = d / (np.linalg.norm(d) + 1e-12)
    return A


def cmat(A, ks):
    M = np.eye(len(ks))
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            if i < j:
                M[i, j] = M[j, i] = float(A[a] @ A[b])
    return M


def greedy(ks, C, m):
    n = len(ks); sel = []; best = np.full(n, -np.inf)
    for _ in range(m):
        g = [-np.inf if j in sel else float(np.mean(np.maximum(best, C[:, j]))) for j in range(n)]
        j = int(np.argmax(g)); sel.append(j); best = np.maximum(best, C[:, j])
    return [ks[j] for j in sel]


def cov_true(ks, C_true, subset):
    idx = [ks.index(k) for k in subset]
    return float(np.mean([max(C_true[k, j] for j in idx) for k in range(len(ks))]))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    z = np.load(os.path.join(ART, f"{ds}_train_resnet50.npz".replace("+", "plus")), allow_pickle=True)
    X = z["X"].astype(np.float64); y = z["y"].astype(int); st = z["subtype"].astype(str)
    Xs = standardize(X)
    zt = np.load(os.path.join(ART, f"axes_{ds}.npz".replace("+", "plus")), allow_pickle=True)
    ks = [str(k) for k in zt["ks"]]; C_true = zt["C"]
    full = {m: greedy(ks, C_true, m) for m in (2, 4, 6, 8)}
    cov_full = {m: cov_true(ks, C_true, full[m]) for m in full}

    print(f"== PILOT: how many images per PAI are needed to decide well? [{ds}] ==")
    print(f"  full-data selection: " + " · ".join(f"m={m}:{cov_full[m]:.3f}" for m in full))
    print(f"\n  {'n/PAI':>6s} {'m':>2s} {'Jaccard vs full':>16s} {'real coverage':>15s} {'loss':>8s}")
    out = {"cov_full": {str(m): cov_full[m] for m in cov_full}, "sizes": {}}
    for n_per in SIZES:
        out["sizes"][str(n_per)] = {}
        for m in (2, 4, 6, 8):
            js, cvs = [], []
            for rep in range(REPS):
                rng = np.random.default_rng(1000 + rep)
                A = axes_from(Xs, y, st, ks, rng, n_per)
                S = greedy(ks, cmat(A, ks), m)
                inter = len(set(S) & set(full[m])); union = len(set(S) | set(full[m]))
                js.append(inter / union)
                cvs.append(cov_true(ks, C_true, S))     # coverage REAL del set elegido con poco dato
            loss = cov_full[m] - float(np.mean(cvs))
            print(f"  {n_per:6d} {m:2d} {np.mean(js):16.2f} {np.mean(cvs):15.3f} {loss:+8.3f}")
            out["sizes"][str(n_per)][str(m)] = {"jaccard": float(np.mean(js)),
                                                "cov_real": float(np.mean(cvs)), "loss": float(loss)}
        print()
    tot = {k: int(np.sum(st == k)) for k in ks}
    print(f"  (actual train size per PAI: min={min(tot.values())} max={max(tot.values())})")
    json.dump(out, open(os.path.join(ART, f"pilot_{ds}.json".replace("+", "plus")), "w"), indent=1)
    print(f"  -> artifacts/pilot_{ds}.json")


if __name__ == "__main__":
    main()
