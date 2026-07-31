#!/usr/bin/env python3
"""
Mapa angular de los PAIs (geometría DAXIS cruda, sin GO/NO-GO) + generación de órdenes O1..O9
y picks de leave-out. Corre en /opt/conda/bin/python (3.13).

Entrada: artifacts/<ds>_train_<bb>.npz (de 00). Canónico: HQ-WMCA resnet50.
Salida:  artifacts/orders.json, artifacts/axes_<ds>.npz, figuras heatmap/dendrograma,
         estabilidad (Kendall tau entre backbones y seeds de shard).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "daxis-paper", "daxis_library"))
import daxis
from daxis_ext import (standardize, pseudo_domains, all_axes, cosine_matrix, aggregate_axis,
                       order_aligned_first, order_greedy, order_spectral, order_zigzag,
                       order_cluster_blocked, pick_diverse)
from scipy.stats import kendalltau

from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa


def load(ds, bb):
    p = os.path.join(ART, f"{ds}_train_{bb}.npz".replace("+", "plus"))
    if not os.path.isfile(p): return None
    z = np.load(p, allow_pickle=True)
    return z["X"].astype(np.float64), z["y"].astype(int), z["subtype"].astype(str), z["subject"].astype(str)


def freq_counts(subtype, y):
    from collections import Counter
    return Counter(subtype[y == 1])


def analyze(ds, bb, shard_seed=0, canonical=False):
    d = load(ds, bb)
    if d is None: return None
    X, y, st, su = d
    Xs = standardize(X)
    axes = all_axes(Xs, y, st)
    ks, C = cosine_matrix(axes)
    out = {"ks": ks, "C": C, "axes": axes}
    # daxis_score canónico (pseudo-dominios con shard live por sujeto) — validez del score global
    if canonical:
        dom, _ = pseudo_domains(y, st, su, seed=shard_seed)
        res = daxis.daxis_score(X, y, dom, mode="binary", n_boot=100, n_perm=100, random_state=0)
        # reordenar la matriz canónica al orden de ks
        idx = [res.domains.index(k) for k in ks]
        out["daxis_score"] = float(res.score); out["daxis_p"] = float(res.p_value)
        out["C_daxis"] = res.matrix[np.ix_(idx, idx)]
        out["ci"] = list(res.ci)
    return out


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    # --- canónico: resnet50, shard seed 0 ---
    R = analyze(ds, "resnet50", 0, canonical=True)
    assert R is not None, "faltan embeddings resnet50 (corre 00 primero)"
    # str() explicito: numpy.str_ no serializa limpio a JSON y rompe a los consumidores
    ks, C, axes = [str(k) for k in R["ks"]], R["C"], {str(k): v for k, v in R["axes"].items()}
    d0 = load(ds, "resnet50"); X, y, st, su = d0; Xs = standardize(X)
    fc = freq_counts(st, y)
    start = max(fc, key=fc.get)                     # el más frecuente (spec usuario)

    orders = {
        "O1_aligned_first": order_aligned_first(ks, C, desc=True),
        "O2_divergent_first": order_aligned_first(ks, C, desc=False),
        "O3_greedy_chain": order_greedy(ks, axes, Xs, y, st, start, opposite=False),
        "O4_greedy_opposite": order_greedy(ks, axes, Xs, y, st, start, opposite=True),
        "O5_spectral": order_spectral(ks, C),
        "O6_cluster_blocked": order_cluster_blocked(ks, C),
        "O9_zigzag": order_zigzag(ks, axes, Xs, y, st, start),
    }
    # leave-out picks
    off = C.copy(); np.fill_diagonal(off, -2)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    red_pair = sorted([ks[i], ks[j]], key=lambda k: fc[k])       # [menor, mayor]
    mean_cos = (C.sum(1) - 1.0) / (len(ks) - 1)
    outlier = ks[int(np.argmin(mean_cos))]
    aligned_rank = orders["O1_aligned_first"]
    picks = {
        "L1_redundant_drop": red_pair[0], "L1_redundant_pair": red_pair,
        "L1_pair_cos": float(C[i, j]),
        "L2_outlier_drop": outlier,
        "L3_topm_aligned": {str(m): aligned_rank[:m] for m in (2, 4, 6, 8)},
        "L3b_diverse": {str(m): pick_diverse(ks, C, m) for m in (2, 4, 6, 8)},
        "start_most_frequent": start,
    }
    # estabilidad: Kendall tau del ranking O1 entre variantes (backbone / shard-seed no afecta
    # a los ejes con mu_live global, pero sí al C_daxis; comparamos ambos caminos)
    stab = {}
    o1_c = orders["O1_aligned_first"]
    if "C_daxis" in R:
        o1_daxis = order_aligned_first(ks, R["C_daxis"], desc=True)
        stab["tau_axes_vs_daxismatrix"] = float(kendalltau([o1_c.index(k) for k in ks],
                                                           [o1_daxis.index(k) for k in ks])[0])
    Rx = analyze(ds, "resnext101_64x4d")
    if Rx is not None:
        o1_x = order_aligned_first(Rx["ks"], Rx["C"], desc=True)
        stab["tau_resnet50_vs_resnext101"] = float(kendalltau([o1_c.index(k) for k in ks],
                                                              [o1_x.index(k) for k in ks])[0])
    for sseed in (1, 2):
        Rs = analyze(ds, "resnet50", sseed, canonical=True)
        o1_s = order_aligned_first(Rs["ks"], Rs["C_daxis"], desc=True)
        stab[f"tau_shardseed0_vs_{sseed}"] = float(kendalltau([o1_c.index(k) for k in ks],
                                                              [o1_s.index(k) for k in ks])[0])

    # guardar
    np.savez_compressed(os.path.join(ART, f"axes_{ds}.npz".replace("+", "plus")),
                        ks=np.array(ks), C=C, **{f"axis_{k}": axes[k] for k in ks})
    orders = {k: [str(x) for x in v] for k, v in orders.items()}
    picks = json.loads(json.dumps(picks, default=str))
    blob = {"dataset": ds, "orders": orders, "picks": picks, "stability": stab,
            "daxis_score": R.get("daxis_score"), "daxis_p": R.get("daxis_p"), "ci": R.get("ci"),
            "freq": {k: int(fc[k]) for k in ks},
            "mean_cos": {k: float(mean_cos[ii]) for ii, k in enumerate(ks)}}
    op = os.path.join(ART, "orders.json")
    allo = json.load(open(op)) if os.path.isfile(op) else {}
    allo[ds] = blob
    json.dump(allo, open(op, "w"), indent=1, ensure_ascii=False)

    # figuras
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        im = axs[0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        axs[0].set_xticks(range(len(ks))); axs[0].set_xticklabels(ks, rotation=75, fontsize=8)
        axs[0].set_yticks(range(len(ks))); axs[0].set_yticklabels(ks, fontsize=8)
        axs[0].set_title(f"{ds}: cosenos entre ejes discriminantes"); fig.colorbar(im, ax=axs[0])
        D = 1 - C; np.fill_diagonal(D, 0)
        dendrogram(linkage(squareform(D, checks=False), method="average"), labels=ks, ax=axs[1],
                   leaf_rotation=75)
        axs[1].set_title("dendrograma angular")
        fig.tight_layout(); fig.savefig(os.path.join(ART, f"map_{ds}.png".replace("+", "plus")), dpi=130)
    except Exception as e:
        print(f"(figura no generada: {e})")

    print(f"== {ds} ==")
    print(f"daxis_score={blob['daxis_score']} p={blob['daxis_p']}")
    print(f"mean_cos: " + " ".join(f"{k}:{blob['mean_cos'][k]:+.2f}" for k in ks))
    for k, v in orders.items(): print(f"  {k}: {v}")
    for k, v in picks.items(): print(f"  {k}: {v}")
    print(f"estabilidad: {stab}")


if __name__ == "__main__":
    main()
