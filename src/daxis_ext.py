#!/usr/bin/env python3
"""
Geometry helpers built on the DAXIS discriminant-axis construction.

The upstream daxis library exposes only a scalar score, and its own repo forbids editing, so the
pieces this paper needs live here: the per-group axis vectors themselves, the cosine matrix
between them, the orderings derived from that matrix, and per-sample axial scores.

Core definition. For attack type k, the discriminant axis is the unit vector joining the class
means in a frozen feature space:

    d_k = normalise( mean(features of attack k) - mean(features of bonafide) )

Everything else is angles between those vectors. Only MEANS are estimated, which is far more
stable in high dimension than an inverse covariance.
"""
import numpy as np


# ---------- standardisation (pooled, as in DAXIS: once over EVERYTHING) ----------

def standardize(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-12
    return (X - mu) / sd


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ---------- pseudo-dominios ----------

def live_shards(subject, live_mask, subtypes, seed=0):
    """Asigna cada SUJETO bonafide a un subtipo (round-robin seeded) -> shards disjuntos
    by subject. Returns a domain array: the subtype for spoof rows, and the shard's subtype for bonafide."""
    rng = np.random.RandomState(seed)
    subs = np.unique(subject[live_mask])
    order = rng.permutation(len(subs))
    assign = {s: subtypes[i % len(subtypes)] for i, s in enumerate(subs[order])}
    return assign


def pseudo_domains(y, subtype, subject, seed=0):
    """domain[i] = the spoof subtype, or the subtype assigned to a bonafide subject (shard)."""
    live = (y == 0)
    sts = sorted(set(subtype[~live]))
    assign = live_shards(subject, live, sts, seed)
    dom = np.array([subtype[i] if not live[i] else assign[subject[i]] for i in range(len(y))])
    return dom, sts


# ---------- axes ----------

def axis_of(Xs, y, mask_spoof_group, live_mask):
    """Group axis using a GLOBAL bonafide mean (a stable estimator for ordering/greedy).
    The canonical daxis_score uses a per-shard bonafide mean; step 01 compares both."""
    return _unit(Xs[mask_spoof_group].mean(0) - Xs[live_mask].mean(0))


def all_axes(Xs, y, subtype):
    """dict subtipo -> eje (mu_live global)."""
    live = (y == 0)
    return {st: axis_of(Xs, y, (subtype == st) & ~live, live) for st in sorted(set(subtype[~live]))}


def aggregate_axis(Xs, y, subtype, group):
    """Axis of a SET of subtypes (their pooled spoof samples against global bonafide)."""
    live = (y == 0)
    m = np.isin(subtype, list(group)) & ~live
    return _unit(Xs[m].mean(0) - Xs[live].mean(0))


def cosine_matrix(axes, order=None):
    ks = order or sorted(axes)
    D = np.stack([axes[k] for k in ks])
    return ks, np.clip(D @ D.T, -1, 1)


# ---------- orderings derived from the geometry ----------

def order_aligned_first(ks, C, desc=True):
    mean_cos = (C.sum(1) - 1.0) / (len(ks) - 1)          # coseno medio al resto (sin diagonal)
    idx = np.argsort(-mean_cos if desc else mean_cos)
    return [ks[i] for i in idx]


def order_greedy(ks, axes, Xs, y, subtype, start, opposite=False):
    """Starts at `start`; each step adds the subtype with the highest (or lowest) cosine
    against the AGGREGATE axis of what is already selected (recomputed over the union)."""
    left = [k for k in ks if k != start]
    seq = [start]
    while left:
        agg = aggregate_axis(Xs, y, subtype, seq)
        cos = np.array([float(axes[k] @ agg) for k in left])
        i = int(np.argmin(cos)) if opposite else int(np.argmax(cos))
        seq.append(left.pop(i))
    return seq


def order_spectral(ks, C):
    """Barrido angular suave: orden por el 2º autovector (Fiedler) del laplaciano de afinidad."""
    A = (C + 1) / 2.0                                     # afinidad en [0,1]
    d = A.sum(1)
    L = np.diag(d) - A
    w, V = np.linalg.eigh(L)
    f = V[:, 1]                                           # Fiedler
    return [ks[i] for i in np.argsort(f)]


def order_zigzag(ks, axes, Xs, y, subtype, start):
    """Alterna cerca/lejos angular respecto al agregado actual."""
    left = [k for k in ks if k != start]
    seq = [start]; near = True
    while left:
        agg = aggregate_axis(Xs, y, subtype, seq)
        cos = np.array([float(axes[k] @ agg) for k in left])
        i = int(np.argmax(cos)) if near else int(np.argmin(cos))
        seq.append(left.pop(i)); near = not near
    return seq


def order_cluster_blocked(ks, C, n_blocks=3):
    """Dendrograma angular (average linkage sobre 1-cos) -> bloques; bloques por
    alineamiento medio descendente; dentro del bloque, por coseno medio."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1 - C; np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), n_blocks, criterion="maxclust")
    mean_cos = (C.sum(1) - 1.0) / (len(ks) - 1)
    blocks = {}
    for i, k in enumerate(ks): blocks.setdefault(lab[i], []).append(i)
    ordered_blocks = sorted(blocks.values(), key=lambda ix: -np.mean(mean_cos[ix]))
    return [ks[i] for blk in ordered_blocks for i in sorted(blk, key=lambda i: -mean_cos[i])]


def pick_diverse(ks, C, m):
    """Farthest-point sampling angular: subset de m subtipos maximizando diversidad."""
    mean_cos = (C.sum(1) - 1.0) / (len(ks) - 1)
    sel = [int(np.argmin(mean_cos))]                      # seed: the most distant one
    while len(sel) < m:
        cand = [i for i in range(len(ks)) if i not in sel]
        # maximise the minimum angle (= minimise the maximum cosine to what is chosen)
        best = min(cand, key=lambda i: max(C[i, j] for j in sel))
        sel.append(best)
    return [ks[i] for i in sel]


# ---------- per-sample scores (new extension) ----------

def sample_scores(Xs, y, subtype):
    """S1 margen axial, S2 delta LOO de alineamiento, S3 coherencia local.
    Xs = features YA estandarizadas (pooled). Devuelve dict de arrays (n,)."""
    live = (y == 0)
    mu_live = Xs[live].mean(0)
    sts = sorted(set(subtype[~live]))
    axes = all_axes(Xs, y, subtype)
    # midpoint por grupo para el margen firmado
    S1 = np.zeros(len(y)); S2 = np.zeros(len(y)); S3 = np.full(len(y), np.nan)
    # per-group "rest" axis (mean of the other axes), used by S2
    rest = {st: _unit(np.mean([axes[o] for o in sts if o != st], axis=0)) for st in sts}
    for st in sts:
        d = axes[st]
        g_sp = (subtype == st) & ~live
        mu_sp = Xs[g_sp].mean(0); n_sp = int(g_sp.sum())
        c = (mu_sp + mu_live) / 2.0
        # S1 para spoof del grupo y para live (live usa el axis de su... todos los axes? usar
        # the worst margin over all axes would be too harsh; we use the group axis for spoof rows
        # y para live el margen respecto al axis AGREGADO global)
        S1[g_sp] = (Xs[g_sp] - c) @ d                      # y=1: lado positivo = correcto
        S3[g_sp] = ((Xs[g_sp] - mu_live) / (np.linalg.norm(Xs[g_sp] - mu_live, axis=1, keepdims=True) + 1e-12)) @ d
        # S2: quitar x del grupo -> nuevo axis (update de media en forma cerrada)
        base = float(d @ rest[st])
        Xg = Xs[g_sp]
        mu_wo = (mu_sp[None, :] * n_sp - Xg) / (n_sp - 1)          # (n_g, p)
        d_wo = mu_wo - mu_live[None, :]
        d_wo /= (np.linalg.norm(d_wo, axis=1, keepdims=True) + 1e-12)
        S2[g_sp] = d_wo @ rest[st] - base                  # >0: quitarla MEJORA el alineamiento
    # live: margen respecto al axis agregado global (lado negativo = correcto para y=0)
    agg = aggregate_axis(Xs, y, subtype, sts)
    c_all = (Xs[~live].mean(0) + mu_live) / 2.0
    S1[live] = -((Xs[live] - c_all) @ agg)                 # firmado: positivo = lado correcto
    return {"S1": S1, "S2": S2, "S3": S3}
