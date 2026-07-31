#!/usr/bin/env python3
"""
FIGURAS DE GEOMETRÍA PARA EL PAPER — PCA, t-SNE y DBSCAN.

Objetivo doble:
 (a) mostrar visualmente la estructura angular de los PAIs que gobierna la ley de cobertura;
 (b) CONTRASTE que justifica el método: ¿esa estructura se ve en el espacio de features crudo
     (PCA / t-SNE / DBSCAN no supervisados), o hace falta la geometría DISCRIMINANTE (los ejes)?
     Si el clustering no supervisado NO recupera los PAIs pero los ejes sí los ordenan, entonces
     la parte "label-aware" del método no es decorativa. Se cuantifica con ARI/NMI.

Paneles:
  1. PCA 2D de las muestras, coloreadas por PAI            (¿se separan solos?)
  2. t-SNE 2D de las muestras, coloreadas por PAI
  3. DBSCAN sobre las muestras -> ARI/NMI vs PAI real      (clustering no supervisado)
  4. PCA 2D de los EJES discriminantes (la geometría DAXIS)
  5. DBSCAN sobre distancia ANGULAR entre ejes (1-cos, precomputed) -> familias de PAIs
  6. Dendrograma angular + cobertura marginal por PAI

Uso: /opt/conda/bin/python 11_geometry_viz.py [HQ-WMCA] [n_muestras_tsne]
Salida: artifacts/geom_<ds>.png + artifacts/geom_<ds>.json (métricas)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daxis_ext import standardize, all_axes, cosine_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa


def auto_eps(X, k=10, q=90):
    """eps de DBSCAN por el codo de la distancia al k-ésimo vecino (heurística estándar)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.percentile(d[:, -1], q))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    n_tsne = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    z = np.load(os.path.join(ART, f"{ds}_train_resnet50.npz".replace("+", "plus")), allow_pickle=True)
    X = z["X"].astype(np.float64); y = z["y"].astype(int); st = z["subtype"].astype(str)
    Xs = standardize(X)

    # ejes discriminantes + matriz de cosenos (la geometría del método)
    A = all_axes(Xs, y, st)
    ks, C = cosine_matrix(A)
    ks = [str(k) for k in ks]
    D_ang = np.clip(1.0 - C, 0, 2); np.fill_diagonal(D_ang, 0.0)

    # submuestra equilibrada por PAI para PCA/t-SNE/DBSCAN
    rng = np.random.default_rng(0)
    groups = ["live"] + ks
    per = max(50, n_tsne // len(groups))
    idx = []
    for g in groups:
        gi = np.where(st == g)[0]
        idx += list(rng.choice(gi, min(len(gi), per), replace=False))
    idx = np.array(idx)
    Xsub, stsub = Xs[idx], st[idx]
    lab_true = np.array([groups.index(s) for s in stsub])

    out = {"dataset": ds, "n_sub": int(len(idx)), "n_groups": len(groups)}

    # --- espacio de features: PCA / t-SNE / DBSCAN (no supervisado) ---
    P = PCA(n_components=50, random_state=0).fit_transform(Xsub)
    P2 = P[:, :2]
    T2 = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(P)
    eps = auto_eps(P[:, :10])
    db = DBSCAN(eps=eps, min_samples=10).fit(P[:, :10])
    out["dbscan_samples"] = {
        "eps": eps, "n_clusters": int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)),
        "noise_frac": float(np.mean(db.labels_ == -1)),
        "ARI_vs_PAI": float(adjusted_rand_score(lab_true, db.labels_)),
        "NMI_vs_PAI": float(normalized_mutual_info_score(lab_true, db.labels_)),
    }
    # ¿al menos separa live vs spoof?
    out["dbscan_samples"]["ARI_vs_binary"] = float(
        adjusted_rand_score((stsub != "live").astype(int), db.labels_))

    # --- espacio de EJES: PCA de los ejes + DBSCAN angular ---
    Amat = np.stack([A[k] for k in ks])
    E2 = PCA(n_components=2, random_state=0).fit_transform(Amat)
    best = None
    for e in np.arange(0.30, 1.00, 0.05):
        lb = DBSCAN(eps=float(e), min_samples=2, metric="precomputed").fit(D_ang).labels_
        nc = len(set(lb)) - (1 if -1 in lb else 0)
        if nc >= 2:
            try:
                s = silhouette_score(D_ang, lb, metric="precomputed")
            except Exception:
                s = -1
            if best is None or s > best[0]: best = (s, float(e), lb, nc)
    if best:
        sil, eps_ax, lab_ax, ncl = best
        fam = {}
        for k, l in zip(ks, lab_ax): fam.setdefault(int(l), []).append(k)
        out["dbscan_axes"] = {"eps": eps_ax, "silhouette": float(sil), "n_families": ncl,
                              "families": {str(a): b for a, b in fam.items()}}
    else:
        lab_ax = np.zeros(len(ks), int); out["dbscan_axes"] = {"n_families": 1}

    mean_cos = (C.sum(1) - 1.0) / (len(ks) - 1)

    # ---------------- figura ----------------
    fig = plt.figure(figsize=(16, 9.5))
    cmap = plt.get_cmap("tab20")
    col = {g: cmap(i % 20) for i, g in enumerate(groups)}

    ax = fig.add_subplot(2, 3, 1)
    for g in groups:
        m = stsub == g
        ax.scatter(P2[m, 0], P2[m, 1], s=6, alpha=.6, color=col[g], label=g, edgecolors="none")
    ax.set_title("1. PCA de las muestras (por PAI)"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=2, ncol=2, loc="best")

    ax = fig.add_subplot(2, 3, 2)
    for g in groups:
        m = stsub == g
        ax.scatter(T2[m, 0], T2[m, 1], s=6, alpha=.6, color=col[g], edgecolors="none")
    ax.set_title("2. t-SNE de las muestras (por PAI)"); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(2, 3, 3)
    lb = db.labels_
    ax.scatter(T2[lb == -1, 0], T2[lb == -1, 1], s=6, c="lightgray", label="ruido", edgecolors="none")
    for c in sorted(set(lb) - {-1}):
        m = lb == c
        ax.scatter(T2[m, 0], T2[m, 1], s=6, alpha=.7, edgecolors="none")
    d = out["dbscan_samples"]
    ax.set_title(f"3. DBSCAN no supervisado\nARI vs PAI={d['ARI_vs_PAI']:.2f} · NMI={d['NMI_vs_PAI']:.2f}"
                 f" · {d['n_clusters']} clusters", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(2, 3, 4)
    for i, k in enumerate(ks):
        ax.scatter(E2[i, 0], E2[i, 1], s=90, color=col[k], edgecolors="k", linewidths=.5)
        ax.annotate(k, (E2[i, 0], E2[i, 1]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_title("4. PCA de los EJES discriminantes\n(la geometría DAXIS)", fontsize=10)
    ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 3, 5)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    order = np.argsort(lab_ax)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=80, fontsize=7)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=7)
    fam_txt = " | ".join(",".join(v) for v in out["dbscan_axes"].get("families", {}).values())
    ax.set_title(f"5. cosenos entre ejes · familias DBSCAN:\n{fam_txt}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=.046)

    ax = fig.add_subplot(2, 3, 6)
    dendrogram(linkage(squareform(D_ang, checks=False), method="average"),
               labels=ks, ax=ax, leaf_rotation=80)
    ax.set_title("6. dendrograma angular (1 - cos)", fontsize=10)
    ax.tick_params(labelsize=7)

    fig.suptitle(f"{ds}: la estructura de PAIs vive en los ejes discriminantes, "
                 f"no en el clustering no supervisado del espacio de features", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIG_DIR, f"geom_{ds}.png".replace("+", "plus"))
    fig.savefig(p, dpi=140); plt.close(fig)

    out["mean_cos"] = {k: float(mean_cos[i]) for i, k in enumerate(ks)}
    json.dump(out, open(os.path.join(ART, f"geom_{ds}.json".replace("+", "plus")), "w"), indent=1)

    print(f"== geometría [{ds}] · {len(idx)} muestras, {len(groups)} grupos ==")
    print(f"  DBSCAN no supervisado (features): {d['n_clusters']} clusters, ruido {d['noise_frac']:.0%}")
    print(f"    ARI vs PAI      = {d['ARI_vs_PAI']:+.3f}   <- ¿recupera los tipos de ataque?")
    print(f"    NMI vs PAI      = {d['NMI_vs_PAI']:+.3f}")
    print(f"    ARI vs live/spoof = {d['ARI_vs_binary']:+.3f}")
    da = out["dbscan_axes"]
    print(f"  DBSCAN angular (ejes): {da.get('n_families')} familias, silhouette={da.get('silhouette', float('nan')):.3f}")
    for a, b in da.get("families", {}).items(): print(f"    familia {a}: {', '.join(b)}")
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
