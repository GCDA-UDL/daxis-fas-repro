#!/usr/bin/env python3
"""
AXIAL COORDINATES - is the DAXIS geometry useful as a REPRESENTATION, not just a diagnostic?

Construction: project each sample onto the per-PAI discriminant axes,
    z_i = [<x_i, d_1>, ..., <x_i, d_K>]
From 2048 dims down to K, in a space where each coordinate is "how much does this sample look like
attack k rather than bonafide".

Does it add anything over what is known? The construction is close to LDA and
to nearest-class-mean, so the FAIR comparison is against LDA (same dims, also supervised) and
against a RANDOM projection of K dims (unsupervised control, same dimensionality). Hypothesis:
the axes need only MEANS (stable in high dimension) whereas LDA needs a 2048x2048 inverse
covariance (ill-conditioned with few samples), so the axes should hold up better.

Anti-circularity protocol: axes and LDA are fitted on a TRAIN split and evaluated on DISJOINT
SUBJECTS. Without that, "a label-derived space recovers the labels" is a tautology.

Result: the hypothesis turned out false with full data - LDA wins on everything. The axes only win on the
binary task in the small-sample regime. Reported as a footnote, not a contribution.

Usage: python 12_axial_space.py [HQ-WMCA]
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
SUBSAMPLE_SWEEP = [10, 25, 50, 100, 200, None]     # samples/PAI used to estimate the representation


def axial_fit(Xtr, ytr, sttr, ks):
    """DAXIS axes: one direction per PAI (spoof_k vs bonafide). Means only."""
    mu_live = Xtr[ytr == 0].mean(0)
    W = []
    for k in ks:
        m = (sttr == k) & (ytr == 1)
        if m.sum() == 0: W.append(np.zeros(Xtr.shape[1])); continue
        d = Xtr[m].mean(0) - mu_live
        W.append(d / (np.linalg.norm(d) + 1e-12))
    return np.stack(W).T                            # (p, K)


def eval_rep(Ztr, Zte, sttr, stte, yte, tag, out):
    """Evaluates a representation on the held-out split."""
    r = {}
    # 1. kNN on PAI (does it preserve attack identity?)
    kn = KNeighborsClassifier(n_neighbors=15).fit(Ztr, sttr)
    r["knn_pai_balacc"] = float(balanced_accuracy_score(stte, kn.predict(Zte)))
    # 2. sonda lineal live/spoof
    lr = LogisticRegression(max_iter=2000).fit(Ztr, (sttr != "live").astype(int))
    r["probe_auc_livespoof"] = float(roc_auc_score(yte, lr.decision_function(Zte)))
    # 3. unsupervised DBSCAN on the held-out split: does it recover the PAIs?
    Zn = (Zte - Zte.mean(0)) / (Zte.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=10).fit(Zn)
    d, _ = nn.kneighbors(Zn)
    eps = float(np.percentile(d[:, -1], 90))
    lb = DBSCAN(eps=eps, min_samples=10).fit(Zn).labels_
    r["dbscan_ari_pai"] = float(adjusted_rand_score(stte, lb))
    r["dbscan_nmi_pai"] = float(normalized_mutual_info_score(stte, lb))
    r["dbscan_nclusters"] = int(len(set(lb)) - (1 if -1 in lb else 0))
    # 4. silhouette of the true partition (is the geometry well formed?)
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

    # --- split por subject (anti-circularidad y anti-fuga) ---
    subs = sorted(set(su)); rng = np.random.default_rng(0); rng.shuffle(subs)
    n_te = max(1, int(len(subs) * 0.3))
    te_subs = set(subs[:n_te])
    m_te = np.array([s in te_subs for s in su]); m_tr = ~m_te
    Xs = standardize(X)                                   # global standardisation (shared by all)
    Xtr, Xte = Xs[m_tr], Xs[m_te]
    ytr, yte = y[m_tr], y[m_te]
    sttr, stte = st[m_tr], st[m_te]
    print(f"== AXIAL COORDINATES [{ds}] ==")
    print(f"  train {m_tr.sum()} samples / {len(subs)-len(te_subs)} subjects · "
          f"held-out {m_te.sum()} / {len(te_subs)} subjects · K={len(ks)} PAIs")
    # every class must exist on both sides for the metrics to be meaningful
    faltan = [k for k in ks if (sttr == k).sum() == 0 or (stte == k).sum() == 0]
    if faltan: print(f"  warning: PAIs missing from a split: {faltan}")

    out = {"dataset": ds, "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
           "K": len(ks), "reps": {}}
    K = len(ks)
    print(f"\n  --- representaciones (ajustadas en train, evaluadas en subjects NO VISTOS) ---")
    reps = out["reps"]
    # baseline: raw features reduced by PCA (unsupervised) to 50 and to K dims
    p50 = PCA(n_components=50, random_state=0).fit(Xtr)
    eval_rep(p50.transform(Xtr), p50.transform(Xte), sttr, stte, yte, "PCA-50 (unsup.)", reps)
    pk = PCA(n_components=K, random_state=0).fit(Xtr)
    eval_rep(pk.transform(Xtr), pk.transform(Xte), sttr, stte, yte, f"PCA-{K} (unsup.)", reps)
    # dimensionality control: random projection to K dims
    grp = GaussianRandomProjection(n_components=K, random_state=0).fit(Xtr)
    eval_rep(grp.transform(Xtr), grp.transform(Xte), sttr, stte, yte, f"random-{K} (control)", reps)
    # supervised LDA (same label information, needs Sigma^-1)
    try:
        lda = LinearDiscriminantAnalysis(n_components=min(K, len(set(sttr)) - 1)).fit(Xtr, sttr)
        eval_rep(lda.transform(Xtr), lda.transform(Xte), sttr, stte, yte, "LDA (sup., Sigma^-1)", reps)
    except Exception as e:
        print(f"  LDA failed: {e}")
    # DAXIS axial coordinates (means only)
    W = axial_fit(Xtr, ytr, sttr, ks)
    eval_rep(Xtr @ W, Xte @ W, sttr, stte, yte, "AXIAL DAXIS (means only)", reps)

    # --- small-sample regime: does it hold up with few samples per PAI? ---
    print(f"\n  --- small-sample sweep (samples/PAI used to FIT the representation) ---")
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
        lab = "all" if n_per is None else str(n_per)
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
                                          "t-SNE over features (PCA-50)"),
                                         ((Xte @ W)[sel], f"t-SNE sobre AXIAL COORDINATES ({K}d)")]):
            E = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Z)
            for g in groups:
                m = stte[sel] == g
                ax.scatter(E[m, 0], E[m, 1], s=7, alpha=.65, color=col[g], label=g, edgecolors="none")
            ax.set_title(tt, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        axs[0].legend(fontsize=6, markerscale=2, ncol=2)
        labs = list(sweep); ax = axs[2]
        ax.plot(range(len(labs)), [sweep[l]["axial_knn"] for l in labs], "o-", label="AXIAL (means)")
        ax.plot(range(len(labs)), [sweep[l]["lda_knn"] for l in labs], "s--", label="LDA (Sigma^-1)")
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.set_xlabel("samples/PAI used to fit"); ax.set_ylabel("kNN-PAI bal.acc (held-out)")
        ax.set_title("small-sample regime", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.suptitle(f"{ds}: coordenadas axiales vs features vs LDA (subjects no vistos)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(FIG_DIR, f"axial_{ds}.png".replace("+", "plus")), dpi=140)
        print(f"\n  figure -> artifacts/axial_{ds}.png")
    except Exception as e:
        print(f"  (figure not generated: {e})")


if __name__ == "__main__":
    main()
