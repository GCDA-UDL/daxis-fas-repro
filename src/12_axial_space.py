#!/usr/bin/env python3
"""
COORDENADAS AXIALES — ¿la geometría DAXIS sirve como REPRESENTACIÓN, no solo como diagnóstico?

Construcción: con los ejes discriminantes por PAI  d_k = norm(mu_k^spoof - mu^live)  se proyecta
cada muestra a K dimensiones:      z_i = [<x_i, d_1>, ..., <x_i, d_K>]
De 2048 dims a K (=nº de PAIs), en un espacio donde cada coordenada es "cuánto se parece esta
muestra al ataque k frente a bonafide".

Pregunta honesta: ¿aporta algo sobre lo ya conocido? La construcción está cerca de LDA y de
nearest-class-mean, así que la comparación JUSTA es contra LDA (mismas dims, también supervisada)
y contra una proyección ALEATORIA de K dims (control no supervisado, misma dimensionalidad).
Hipótesis: los ejes solo necesitan MEDIAS (robustas en alta dimensión) mientras LDA necesita
Sigma^-1 de 2048x2048 (mal condicionada con pocas muestras) -> los ejes deberían aguantar mejor.

Protocolo anti-circularidad: los ejes/LDA se ajustan en un split de TRAIN y se evalúan en
SUJETOS DISJUNTOS. Sin esto, "el espacio derivado de las etiquetas recupera las etiquetas" es
una tautología.

Métricas en held-out: kNN(PAI) balanced accuracy · DBSCAN ARI/NMI vs PAI · AUC live/spoof con
sonda lineal · silhouette. Además barrido de nº de muestras/PAI para ver el régimen small-sample.

Uso: /opt/conda/bin/python 12_axial_space.py [HQ-WMCA]
Salida: artifacts/axial_<ds>.json + artifacts/axial_<ds>.png
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daxis_ext import standardize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             balanced_accuracy_score, roc_auc_score, silhouette_score)
from sklearn.random_projection import GaussianRandomProjection

from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
SUBSAMPLE_SWEEP = [10, 25, 50, 100, 200, None]     # muestras/PAI para estimar la representación


def axial_fit(Xtr, ytr, sttr, ks):
    """Ejes DAXIS: una dirección por PAI (spoof_k vs live). Solo medias."""
    mu_live = Xtr[ytr == 0].mean(0)
    W = []
    for k in ks:
        m = (sttr == k) & (ytr == 1)
        if m.sum() == 0: W.append(np.zeros(Xtr.shape[1])); continue
        d = Xtr[m].mean(0) - mu_live
        W.append(d / (np.linalg.norm(d) + 1e-12))
    return np.stack(W).T                            # (p, K)


def eval_rep(Ztr, Zte, sttr, stte, yte, tag, out):
    """Evalúa una representación en el split held-out."""
    r = {}
    # 1. kNN de PAI (¿preserva la identidad del ataque?)
    kn = KNeighborsClassifier(n_neighbors=15).fit(Ztr, sttr)
    r["knn_pai_balacc"] = float(balanced_accuracy_score(stte, kn.predict(Zte)))
    # 2. sonda lineal live/spoof
    lr = LogisticRegression(max_iter=2000).fit(Ztr, (sttr != "live").astype(int))
    r["probe_auc_livespoof"] = float(roc_auc_score(yte, lr.decision_function(Zte)))
    # 3. DBSCAN no supervisado sobre el held-out -> ¿recupera los PAIs?
    Zn = (Zte - Zte.mean(0)) / (Zte.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=10).fit(Zn)
    d, _ = nn.kneighbors(Zn)
    eps = float(np.percentile(d[:, -1], 90))
    lb = DBSCAN(eps=eps, min_samples=10).fit(Zn).labels_
    r["dbscan_ari_pai"] = float(adjusted_rand_score(stte, lb))
    r["dbscan_nmi_pai"] = float(normalized_mutual_info_score(stte, lb))
    r["dbscan_nclusters"] = int(len(set(lb)) - (1 if -1 in lb else 0))
    # 4. silhouette de la partición verdadera (¿está bien formada la geometría?)
    try:
        r["silhouette_pai"] = float(silhouette_score(Zn, stte))
    except Exception:
        r["silhouette_pai"] = float("nan")
    r["dims"] = int(Zte.shape[1])
    out[tag] = r
    print(f"  {tag:26s} dims={r['dims']:4d}  kNN-PAI={r['knn_pai_balacc']:.3f}  "
          f"probe-AUC={r['probe_auc_livespoof']:.4f}  DBSCAN-ARI={r['dbscan_ari_pai']:+.3f}  "
          f"sil={r['silhouette_pai']:+.3f}")
    return r


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    z = np.load(os.path.join(ART, f"{ds}_train_resnet50.npz".replace("+", "plus")), allow_pickle=True)
    X = z["X"].astype(np.float64); y = z["y"].astype(int)
    st = z["subtype"].astype(str); su = z["subject"].astype(str)
    ks = sorted(set(st[y == 1]))

    # --- split por SUJETO (anti-circularidad y anti-fuga) ---
    subs = sorted(set(su)); rng = np.random.default_rng(0); rng.shuffle(subs)
    n_te = max(1, int(len(subs) * 0.3))
    te_subs = set(subs[:n_te])
    m_te = np.array([s in te_subs for s in su]); m_tr = ~m_te
    Xs = standardize(X)                                   # estandarización global (única para todos)
    Xtr, Xte = Xs[m_tr], Xs[m_te]
    ytr, yte = y[m_tr], y[m_te]
    sttr, stte = st[m_tr], st[m_te]
    print(f"== COORDENADAS AXIALES [{ds}] ==")
    print(f"  train {m_tr.sum()} muestras / {len(subs)-len(te_subs)} sujetos · "
          f"held-out {m_te.sum()} / {len(te_subs)} sujetos · K={len(ks)} PAIs")
    # todas las clases deben existir en ambos lados para que las métricas tengan sentido
    faltan = [k for k in ks if (sttr == k).sum() == 0 or (stte == k).sum() == 0]
    if faltan: print(f"  aviso: PAIs ausentes en un split: {faltan}")

    out = {"dataset": ds, "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
           "K": len(ks), "reps": {}}
    K = len(ks)
    print(f"\n  --- representaciones (ajustadas en train, evaluadas en sujetos NO VISTOS) ---")
    reps = out["reps"]
    # baseline: features crudas reducidas por PCA (no supervisado) a 50 y a K
    p50 = PCA(n_components=50, random_state=0).fit(Xtr)
    eval_rep(p50.transform(Xtr), p50.transform(Xte), sttr, stte, yte, "PCA-50 (no superv.)", reps)
    pk = PCA(n_components=K, random_state=0).fit(Xtr)
    eval_rep(pk.transform(Xtr), pk.transform(Xte), sttr, stte, yte, f"PCA-{K} (no superv.)", reps)
    # control de dimensionalidad: proyección aleatoria a K
    grp = GaussianRandomProjection(n_components=K, random_state=0).fit(Xtr)
    eval_rep(grp.transform(Xtr), grp.transform(Xte), sttr, stte, yte, f"random-{K} (control)", reps)
    # LDA supervisada (misma info de etiquetas, necesita Sigma^-1)
    try:
        lda = LinearDiscriminantAnalysis(n_components=min(K, len(set(sttr)) - 1)).fit(Xtr, sttr)
        eval_rep(lda.transform(Xtr), lda.transform(Xte), sttr, stte, yte, "LDA (superv., Sigma^-1)", reps)
    except Exception as e:
        print(f"  LDA falló: {e}")
    # coordenadas axiales DAXIS (solo medias)
    W = axial_fit(Xtr, ytr, sttr, ks)
    eval_rep(Xtr @ W, Xte @ W, sttr, stte, yte, "AXIAL DAXIS (solo medias)", reps)

    # --- régimen small-sample: ¿aguanta con pocas muestras por PAI? ---
    print(f"\n  --- barrido small-sample (muestras/PAI para AJUSTAR la representación) ---")
    print(f"  {'n/PAI':>6s}  {'AXIAL kNN':>10s} {'LDA kNN':>9s}   {'AXIAL AUC':>10s} {'LDA AUC':>9s}")
    sweep = {}
    for n_per in SUBSAMPLE_SWEEP:
        idx = []
        for g in ["live"] + ks:
            gi = np.where(sttr == g)[0]
            take = len(gi) if n_per is None else min(len(gi), n_per * (2 if g == "live" else 1))
            idx += list(np.random.default_rng(7).choice(gi, take, replace=False))
        idx = np.array(idx)
        Xa, ya, sta = Xtr[idx], ytr[idx], sttr[idx]
        row = {}
        Wn = axial_fit(Xa, ya, sta, ks)
        kn = KNeighborsClassifier(n_neighbors=15).fit(Xa @ Wn, sta)
        row["axial_knn"] = float(balanced_accuracy_score(stte, kn.predict(Xte @ Wn)))
        lr = LogisticRegression(max_iter=2000).fit(Xa @ Wn, (sta != "live").astype(int))
        row["axial_auc"] = float(roc_auc_score(yte, lr.decision_function(Xte @ Wn)))
        try:
            ld = LinearDiscriminantAnalysis(n_components=min(K, len(set(sta)) - 1)).fit(Xa, sta)
            kn2 = KNeighborsClassifier(n_neighbors=15).fit(ld.transform(Xa), sta)
            row["lda_knn"] = float(balanced_accuracy_score(stte, kn2.predict(ld.transform(Xte))))
            lr2 = LogisticRegression(max_iter=2000).fit(ld.transform(Xa), (sta != "live").astype(int))
            row["lda_auc"] = float(roc_auc_score(yte, lr2.decision_function(ld.transform(Xte))))
        except Exception:
            row["lda_knn"] = float("nan"); row["lda_auc"] = float("nan")
        lab = "todas" if n_per is None else str(n_per)
        sweep[lab] = row
        print(f"  {lab:>6s}  {row['axial_knn']:10.3f} {row['lda_knn']:9.3f}   "
              f"{row['axial_auc']:10.4f} {row['lda_auc']:9.4f}")
    out["sweep"] = sweep
    json.dump(out, open(os.path.join(ART, f"axial_{ds}.json".replace("+", "plus")), "w"), indent=1)

    # --- figura ---
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        groups = ["live"] + ks
        cmap = plt.get_cmap("tab20"); col = {g: cmap(i % 20) for i, g in enumerate(groups)}
        rng2 = np.random.default_rng(1)
        sel = []
        for g in groups:
            gi = np.where(stte == g)[0]
            sel += list(rng2.choice(gi, min(len(gi), 220), replace=False)) if len(gi) else []
        sel = np.array(sel)
        for ax, (Z, tt) in zip(axs[:2], [(PCA(n_components=50, random_state=0).fit_transform(Xte[sel]),
                                          "t-SNE sobre features (PCA-50)"),
                                         ((Xte @ W)[sel], f"t-SNE sobre COORDENADAS AXIALES ({K}d)")]):
            E = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Z)
            for g in groups:
                m = stte[sel] == g
                ax.scatter(E[m, 0], E[m, 1], s=7, alpha=.65, color=col[g], label=g, edgecolors="none")
            ax.set_title(tt, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        axs[0].legend(fontsize=6, markerscale=2, ncol=2)
        labs = list(sweep); ax = axs[2]
        ax.plot(range(len(labs)), [sweep[l]["axial_knn"] for l in labs], "o-", label="AXIAL (medias)")
        ax.plot(range(len(labs)), [sweep[l]["lda_knn"] for l in labs], "s--", label="LDA (Sigma^-1)")
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.set_xlabel("muestras/PAI para ajustar"); ax.set_ylabel("kNN-PAI bal.acc (held-out)")
        ax.set_title("régimen small-sample", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.suptitle(f"{ds}: coordenadas axiales vs features vs LDA (sujetos no vistos)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(FIG_DIR, f"axial_{ds}.png".replace("+", "plus")), dpi=140)
        print(f"\n  figura -> artifacts/axial_{ds}.png")
    except Exception as e:
        print(f"  (figura no generada: {e})")


if __name__ == "__main__":
    main()
