#!/usr/bin/env python3
"""
Does the geometry EXPLAIN the catastrophic forgetting measured in the curriculum runs?

Each time a curriculum introduces a new attack type the AUC tends to drop. This correlates, over
the transitions already recorded on disk, the cosine between the incoming axis and the aggregate
direction of what is already trained against the AUC change at that step.

Costs no GPU time: it reads the per-iteration curves that the training runs already wrote.

Usage: python 06_retro_forgetting.py [HQ-WMCA]
"""
import os, sys, csv, glob, json
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daxis_ext import standardize, all_axes, aggregate_axis

from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
CURVES = "/mnt/d2/ibt26/results/curves"


def load_emb(ds, bb="resnet50"):
    p = os.path.join(ART, f"{ds}_train_{bb}.npz".replace("+", "plus"))
    z = np.load(p, allow_pickle=True)
    return z["X"].astype(np.float64), z["y"].astype(int), z["subtype"].astype(str)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    X, y, st = load_emb(ds)
    Xs = standardize(X)
    axes = all_axes(Xs, y, st)

    pts = []   # (cos_entrante_vs_agregado_previo, delta_auc, curva, subtipo, iter)
    pat = os.path.join(CURVES, f"{ds}_within_*IBT*.csv".replace("+", "plus"))
    files = sorted(glob.glob(pat))
    for f in files:
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 3: continue
        seq = []
        for i, r in enumerate(rows):
            sub = r["subtype"]
            try: auc = float(r["test_auc"])
            except Exception: continue
            if sub in ("ALL",) or sub not in axes:
                seq.append((sub, auc)); continue
            if i > 0 and seq:
                prev_subs = [s for s, _ in seq if s in axes]
                if prev_subs:
                    agg = aggregate_axis(Xs, y, st, prev_subs)
                    cos = float(axes[sub] @ agg)
                    d = auc - seq[-1][1]
                    pts.append((cos, d, os.path.basename(f), sub, i + 1))
            seq.append((sub, auc))
    if not pts:
        print(f"sin transitions utilizables (curvas en {CURVES}?)"); return
    cos = np.array([p[0] for p in pts]); dl = np.array([p[1] for p in pts])
    pr = pearsonr(cos, dl); sp = spearmanr(cos, dl)
    print(f"== D3 forgetting vs geometry [{ds}] ==")
    print(f"  n transitions = {len(pts)} (de {len(files)} curvas)")
    print(f"  cos(eje_entrante, agregado_previo) vs ΔAUC:")
    print(f"    Pearson  r={pr[0]:+.3f}  p={pr[1]:.2e}")
    print(f"    Spearman r={sp[0]:+.3f}  p={sp[1]:.2e}")
    print(f"  reading: r>0 means the MORE aligned the incoming PAI, the LESS the AUC drops")
    lo = dl[cos < np.median(cos)]; hi = dl[cos >= np.median(cos)]
    print(f"  ΔAUC medio | poorly aligned entrants: {lo.mean():+.3f}  · well aligned: {hi.mean():+.3f}")
    worst = sorted(pts, key=lambda p: p[1])[:8]
    print("  largest drops:")
    for c, d, f, s, i in worst:
        print(f"    ΔAUC={d:+.3f} cos={c:+.3f} {s:14s} iter{i:2d} {f[:52]}")
    json.dump({"n": len(pts), "pearson_r": pr[0], "pearson_p": pr[1],
               "spearman_r": sp[0], "spearman_p": sp[1],
               "mean_drop_lowcos": float(lo.mean()), "mean_drop_highcos": float(hi.mean()),
               "points": [{"cos": c, "dauc": d, "curve": f, "subtype": s, "iter": i} for c, d, f, s, i in pts]},
              open(os.path.join(ART, f"d3_forgetting_{ds}.json".replace("+", "plus")), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(cos, dl, s=22, alpha=0.7, c="#b2182b")
        z = np.polyfit(cos, dl, 1); xs = np.linspace(cos.min(), cos.max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", lw=1)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("cos(eje del PAI entrante, eje agregado previo)")
        ax.set_ylabel("ΔAUC en la transición")
        ax.set_title(f"{ds}: olvido vs alineamiento angular (r={pr[0]:+.2f})")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"d3_forgetting_{ds}.png".replace("+", "plus")), dpi=130)
        print(f"  figure -> artifacts/d3_forgetting_{ds}.png")
    except Exception as e:
        print(f"  (figure not generated: {e})")


if __name__ == "__main__":
    main()
