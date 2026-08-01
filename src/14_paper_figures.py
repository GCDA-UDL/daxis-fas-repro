#!/usr/bin/env python3
"""
FINAL PAPER FIGURES. Writes into the figures/ directory:
  fig_coverage_law.png   the law across the 7 datasets (grid with regression lines)
  fig_budget.png         selection at a FIXED image budget (4k/8k/16k)
  fig_crossover.png      why 'aligned' wins small and 'diverse' wins large -> it is coverage

Usage: python 14_paper_figures.py
"""
import os, sys, csv, json
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
OUT = FIG_DIR
os.makedirs(OUT, exist_ok=True)
RES_D = RES_DAXIS
RES_M = RES_CURRICULUM


def load(path):
    g = defaultdict(list)
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)): g[r["method"]].append(r)
    return g


G = load(RES_D)


def val_sel(rows):
    if not rows: return None
    mx = max(int(r["iter"]) for r in rows)
    if len(rows) < mx: return None
    b = min(rows, key=lambda r: float(r["dev_eer"]))
    return float(b["auc"]) * 100, float(b["acer"]) * 100


def cells_for(ds, blob):
    pts = []
    for tag, info in blob["cells"].items():
        for s in range(3):
            v = val_sel(G.get(f"std-{tag}-s{s}"))
            if v: pts.append((info["coverage"], info["m"], v[0], v[1]))
    return pts


def hqwmca_cells():
    """HQ-WMCA used the heuristic design (blocks B/F); rebuilt from the saved analysis."""
    d = json.load(open(os.path.join(ART, "coverage_law_HQ-WMCA.json")))
    return [(p["coverage"], p["m"], p["auc"], p["acer"]) for p in d["points"]]


# ---------------------------------------------------------------- fig 1: la ley
def fig_law():
    cells = json.load(open(os.path.join(ART, "coverage_cells.json")))
    data = {"HQ-WMCA": hqwmca_cells()}
    for ds, blob in cells.items():
        p = cells_for(ds, blob)
        if len(p) >= 5: data[ds] = p
    order = ["HQ-WMCA", "CelebA-Spoof", "SiW", "OULU-NPU", "CASIA-FASD", "replay", "CASIA-SURF"]
    order = [d for d in order if d in data]
    n = len(order)
    # COMMON colour normalisation (otherwise each panel uses its own range of m and misleads)
    allm = [p[1] for ds in order for p in data[ds]]
    norm = matplotlib.colors.Normalize(vmin=min(allm), vmax=max(allm))
    fig, axs = plt.subplots(2, n, figsize=(2.9 * n, 6.6), squeeze=False)
    for i, ds in enumerate(order):
        pts = data[ds]
        cov = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
        auc = np.array([p[2] for p in pts]); acer = np.array([p[3] for p in pts])
        rng = cov.max() - cov.min()
        ndist = len(set(np.round(cov, 4)))
        for row, (yv, lab) in enumerate([(auc, "AUC (%)"), (acer, "ACER (%)")]):
            ax = axs[row][i]
            sc = ax.scatter(cov, yv, c=m, cmap="viridis", norm=norm, s=32,
                            edgecolors="k", linewidths=.3)
            # usable range for regression: at least 3 coverage levels and an appreciable span
            if ndist >= 3 and rng > 0.02:
                z = np.polyfit(cov, yv, 1); xs = np.linspace(cov.min(), cov.max(), 20)
                ax.plot(xs, np.polyval(z, xs), "r--", lw=1.3)
                r, pv = pearsonr(cov, yv)
                star = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."
                txt, col = f"$r={r:+.2f}$ {star}", ("black" if pv < .05 else "gray")
            elif ndist >= 2:
                r, pv = pearsonr(cov, yv)
                txt, col = f"$r={r:+.2f}$ (2 levels)", "gray"
            else:
                txt, col = "degenerate range", "gray"
            # annotation INSIDE the axes so it cannot collide with titles or labels
            ax.text(.04, .06 if row == 1 else .93, txt, transform=ax.transAxes,
                    fontsize=8.5, color=col, va="bottom" if row == 1 else "top",
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=1.5))
            if i == 0: ax.set_ylabel(lab, fontsize=9)
            if row == 1: ax.set_xlabel("coverage", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=.25)
        axs[0][i].set_title(ds, fontsize=10, fontweight="bold", pad=6)
    fig.suptitle("Angular coverage predicts PAD performance  "
                 r"$\mathrm{cov}(S)=\mathrm{mean}_k\,\max_{j\in S}\langle d_k,d_j\rangle$",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 0.955, 0.94])
    cax = fig.add_axes([0.965, 0.12, 0.011, 0.72])
    fig.colorbar(sc, cax=cax, label="PAIs in training")
    p = os.path.join(OUT, "fig_coverage_law.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  -> {p}  ({n} datasets)")


# ------------------------------------------------------- fig 2: presupuesto fijo
def fig_budget():
    meta = json.load(open(os.path.join(ART, "budget_cells.json")))["HQ-WMCA"]
    budgets = sorted({m["budget"] for m in meta.values()})
    strat = ["cov", "cb", "wcov", "big", "rand"]
    NAMES = {"cov": "coverage", "cb": "cost-benefit", "wcov": "weighted cov.",
             "big": "largest classes", "rand": "random"}
    COL = {"cov": "#1f77b4", "cb": "#2ca02c", "wcov": "#9467bd", "big": "#ff7f0e", "rand": "#7f7f7f"}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    w = 0.16
    for ax, key, lab in [(axs[0], 2, "AUC (%)"), (axs[1], 3, "ACER (%)")]:
        for si, s in enumerate(strat):
            xs, ys, cvs = [], [], []
            for bi, B in enumerate(budgets):
                tag = next((t for t, mm in meta.items()
                            if mm["budget"] == B and mm["strategy"] == s), None)
                if tag is None: continue
                src = meta[tag].get("duplicate_of", tag)
                vals = [val_sel(G.get(f"std-{src}-s{k}")) for k in range(3)]
                vals = [v for v in vals if v]
                if not vals: continue
                xs.append(bi + (si - 2) * w)
                ys.append(np.mean([v[key - 2] for v in vals]))
                cvs.append(meta[tag]["coverage"])
            ax.bar(xs, ys, width=w, color=COL[s], label=NAMES[s] if key == 2 else None,
                   edgecolor="k", linewidth=.4)
            for x, y, c in zip(xs, ys, cvs):
                ax.text(x, y, f"{c:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b//1000}k imgs" for b in budgets])
        ax.set_ylabel(lab); ax.grid(alpha=.25, axis="y")
        ax.set_ylim(85 if key == 2 else 0, 101 if key == 2 else None)
    axs[0].legend(fontsize=8, ncol=3, loc="lower right")
    fig.suptitle("Selection at a FIXED IMAGE BUDGET (HQ-WMCA, 3 seeds; "
                 "the number above each bar is its coverage)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    p = os.path.join(OUT, "fig_budget.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  -> {p}")


# --------------------------------------------------------- fig 3: el cruce
def fig_crossover():
    picks = json.load(open(os.path.join(ART, "orders.json")))["HQ-WMCA"]["picks"]
    z = np.load(os.path.join(ART, "axes_HQ-WMCA.npz"), allow_pickle=True)
    ks = [str(k) for k in z["ks"]]; C = z["C"]
    cov = lambda S: float(np.mean([max(C[k, ks.index(j)] for j in S) for k in range(len(ks))]))
    ms = [2, 4, 6, 8]
    import re
    def agg(pat):
        v = [val_sel(G.get(k)) for k in G if re.match(pat, k)]
        v = [x for x in v if x]
        return np.mean([x[0] for x in v]) if v else None
    A = [agg(rf"std-L3top{m}A-s") for m in ms]
    D = [agg(rf"std-L3top{m}D-s") for m in ms]
    S = [agg(rf"std-F1sel{m}-s") for m in ms]
    cA = [cov(picks["L3_topm_aligned"][str(m)]) for m in ms]
    cD = [cov(picks["L3b_diverse"][str(m)]) for m in ms]
    cS = [cov(picks["F_daxis_select"][str(m)]) for m in ms]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for y, c, lab, mk in [(A, cA, "aligned", "o"), (D, cD, "diverse", "s"), (S, cS, "max coverage", "^")]:
        axs[0].plot(ms, y, mk + "-", label=lab, lw=1.6)
        axs[1].plot(c, y, mk, ms=9, label=lab)
    axs[0].set_xlabel("PAIs in training ($m$)"); axs[0].set_ylabel("AUC (%)")
    axs[0].set_title("The crossover: aligned wins at small budgets, diverse at large ones", fontsize=10)
    axs[0].set_xticks(ms); axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)
    allc = [x for x in cA + cD + cS]; ally = [x for x in A + D + S if x]
    if len(ally) >= 3:
        zz = np.polyfit([c for c, y in zip(allc, A + D + S) if y], ally, 1)
        xs = np.linspace(min(allc), max(allc), 20)
        axs[1].plot(xs, np.polyval(zz, xs), "r--", lw=1.2)
    axs[1].set_xlabel("subset coverage"); axs[1].set_ylabel("AUC (%)")
    axs[1].set_title("...but read through coverage there is no contradiction", fontsize=10)
    axs[1].grid(alpha=.3); axs[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_crossover.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  -> {p}")


if __name__ == "__main__":
    print("== figuras del paper ==")
    fig_law(); fig_budget(); fig_crossover()
    print("done.")
