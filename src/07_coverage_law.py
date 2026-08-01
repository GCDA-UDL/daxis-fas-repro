#!/usr/bin/env python3
"""
THE ANGULAR COVERAGE LAW - why does 'aligned' win at a small budget and 'diverse' at a large one?

Hypothesis: neither alignment nor dispersion is the causal variable. What predicts performance is
COVERAGE: whether every test class has some trained axis NEARBY.

  cov(S) = mean_k  max_{j in S} cos(d_k, d_j)        (k ranges over all test classes)

This explains the observed crossover: at m=2 the 'diverse' pick (max-min) chooses outliers that
cover the bulk poorly (AUC 81.5); at m=8 dispersion already implies coverage (99.85).

Tested WITHOUT a GPU: every cell of block B (L1/L2/L3, different trained subsets) contributes one
(coverage, measured AUC/ACER) pair, and the law is a regression over those cells.

Usage: python 07_coverage_law.py [HQ-WMCA]
Output: artifacts/coverage_law_<ds>.json plus a figure.
"""
import os, sys, csv, json, re
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
RES = RES_DAXIS
RES_MAIN = RES_CURRICULUM


def load_axes(ds):
    z = np.load(os.path.join(ART, f"axes_{ds}.npz".replace("+", "plus")), allow_pickle=True)
    ks = [str(k) for k in z["ks"]]
    return ks, z["C"]


def val_selected(rows):
    """Honest selection: the iteration with the lowest dev EER."""
    if not rows: return None
    b = min(rows, key=lambda r: float(r["dev_eer"]))
    return float(b["auc"]) * 100, float(b["acer"]) * 100


def coverage(sel_idx, C, all_idx):
    """cov = media sobre TODAS las clases de test del mejor coseno a un eje entrenado."""
    if not sel_idx: return np.nan
    return float(np.mean([max(C[k, j] for j in sel_idx) for k in all_idx]))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    ks, C = load_axes(ds)
    idx = {k: i for i, k in enumerate(ks)}
    all_idx = list(range(len(ks)))
    picks = json.load(open(os.path.join(ART, "orders.json")))[ds]["picks"]

    # --- which subtypes each block-B method label trains on ---
    sets = {}
    for m in (2, 4, 6, 8):
        sets[f"std-L3top{m}A"] = picks["L3_topm_aligned"][str(m)]
        sets[f"std-L3top{m}D"] = picks["L3b_diverse"][str(m)]
        for r in range(3):
            key = f"L3_random{r}"
            if key in picks: sets[f"std-L3top{m}R{r}"] = picks[key][str(m)]
    sets["std-L1drop"] = [k for k in ks if k != picks["L1_redundant_drop"]]
    sets["std-L2drop"] = [k for k in ks if k != picks["L2_outlier_drop"]]

    # --- medir each celda ---
    cells = defaultdict(list)
    for r in csv.DictReader(open(RES)):
        cells[r["method"]].append(r)
    # standard (all classes) from the main campaign, same protocol
    for r in csv.DictReader(open(RES_MAIN)):
        if r["dataset"] == ds and r["model"] == "resnet50" and r["method"].startswith("standard-s"):
            cells[r["method"]].append(r)
    for s in range(5):
        sets[f"standard-s{s}"] = ks[:]     # entrena all

    pts = []
    for meth, rows in cells.items():
        base = meth.rsplit("-s", 1)[0] if "-s" in meth else meth
        train_set = sets.get(base) or sets.get(meth)
        if train_set is None: continue
        mx = max(int(r["iter"]) for r in rows)
        if len(rows) < mx: continue                     # celda incompleta
        v = val_selected(rows)
        if v is None: continue
        sel_idx = [idx[k] for k in train_set if k in idx]
        cov = coverage(sel_idx, C, all_idx)
        mean_cos = float(np.mean([C[i, j] for i in sel_idx for j in sel_idx if i != j])) if len(sel_idx) > 1 else 1.0
        pts.append({"method": meth, "base": base, "m": len(sel_idx), "coverage": cov,
                    "mean_cos_within": mean_cos, "auc": v[0], "acer": v[1]})

    if len(pts) < 5:
        print(f"solo {len(pts)} cells completas; espera a que avance el bloque B"); return

    cov = np.array([p["coverage"] for p in pts]); auc = np.array([p["auc"] for p in pts])
    acer = np.array([p["acer"] for p in pts]); mm = np.array([p["m"] for p in pts])
    within = np.array([p["mean_cos_within"] for p in pts])

    out = {"n_cells": len(pts)}
    print(f"== COVERAGE LAW [{ds}] · {len(pts)} cells ==")
    for name, x, yv, ylab in [("coverage -> AUC", cov, auc, "AUC"),
                              ("coverage -> ACER", cov, acer, "ACER"),
                              ("n classes m -> AUC", mm, auc, "AUC"),
                              ("within-set alignment -> AUC", within, auc, "AUC")]:
        r, p = pearsonr(x, yv); rs, ps = spearmanr(x, yv)
        out[name] = {"pearson_r": float(r), "pearson_p": float(p),
                     "spearman_r": float(rs), "spearman_p": float(ps)}
        print(f"  {name:26s} Pearson r={r:+.3f} (p={p:.2e})   Spearman r={rs:+.3f}")

    # key control: does coverage add MORE than the class count alone?
    try:
        import numpy.linalg as la
        A = np.column_stack([np.ones(len(pts)), mm, cov])
        beta, *_ = la.lstsq(A, auc, rcond=None)
        pred = A @ beta
        ss = 1 - ((auc - pred) ** 2).sum() / ((auc - auc.mean()) ** 2).sum()
        A0 = np.column_stack([np.ones(len(pts)), mm])
        b0, *_ = la.lstsq(A0, auc, rcond=None)
        ss0 = 1 - ((auc - A0 @ b0) ** 2).sum() / ((auc - auc.mean()) ** 2).sum()
        out["r2_m_only"] = float(ss0); out["r2_m_plus_cov"] = float(ss)
        out["beta_coverage"] = float(beta[2])
        print(f"  R2(m only)={ss0:.3f}  ->  R2(m + coverage)={ss:.3f}   (beta_cov={beta[2]:+.2f})")
    except Exception as e:
        print(f"  (regression unavailable: {e})")

    print("\n  cells (m, coverage, AUC, ACER):")
    for p in sorted(pts, key=lambda p: (p["m"], p["base"])):
        print(f"    {p['base']:16s} m={p['m']:2d} cov={p['coverage']:+.3f} AUC={p['auc']:6.2f} ACER={p['acer']:5.2f}")

    out["points"] = pts
    json.dump(out, open(os.path.join(ART, f"coverage_law_{ds}.json".replace("+", "plus")), "w"), indent=1)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, yv, lab in [(axs[0], auc, "AUC test (%)"), (axs[1], acer, "ACER test (%)")]:
            sc = ax.scatter(cov, yv, c=mm, cmap="viridis", s=55, edgecolors="k", linewidths=.4)
            ax.set_xlabel("angular coverage  mean_k max_{j in S} cos(d_k,d_j)"); ax.set_ylabel(lab)
            ax.grid(alpha=.3)
            fig.colorbar(sc, ax=ax, label="PAIs in training")
        r, p = pearsonr(cov, auc)
        axs[0].set_title(f"{ds}: coverage predicts performance (r={r:+.2f}, p={p:.1e})")
        axs[1].set_title("same axis, on ACER")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"coverage_law_{ds}.png".replace("+", "plus")), dpi=130)
        print(f"\n  figure -> artifacts/coverage_law_{ds}.png")
    except Exception as e:
        print(f"  (figure not generated: {e})")


if __name__ == "__main__":
    main()
