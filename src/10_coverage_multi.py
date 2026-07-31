#!/usr/bin/env python3
"""
REPLICACIÓN MULTI-DATASET de la ley de cobertura (diseño de REGRESIÓN).

07_coverage_law encontró en HQ-WMCA que cov(S) = mean_k max_{j∈S} cos(d_k,d_j) predice el
rendimiento (r=+0.795 AUC / -0.856 ACER; R2 0.47->0.77 sobre m). Un dataset no basta para el paper.

Diseño: en vez de comparar heurísticas (alineado/diverso/random), que confunden la variable de
interés, se MUESTREAN SUBCONJUNTOS del train de tamaños variados. Cada subconjunto es un punto
(cobertura, AUC) y la ley se contrasta por regresión. Es más limpio y más barato.

Nota sobre saturación: que un dataset dé AUC~100 con todos los PAIs NO impide el test — al
restringir el train a pocos PAIs el rendimiento cae (en HQ-WMCA, de 99.9 a 81-94), que es
justo la varianza que la regresión necesita.

Uso: /opt/conda/bin/python 10_coverage_multi.py <dataset> [n_subsets] [seeds]
Genera manifiestos + jobs anexados a artifacts/jobs.json (bloque G-cov<ds>).
"""
import os, sys, csv, json, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daxis_ext import standardize, all_axes, cosine_matrix
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
MAN_OUT = os.path.join(HERE, "manifests")
MAN_SRC = os.path.join(HERE, "..", "fas_benchmark", "manifests")
os.makedirs(MAN_OUT, exist_ok=True)
# minutos/job estimados: escala con el nº de clases y el tamaño del dataset
EST = {"CelebA-Spoof": 120, "SiW": 70, "OULU-NPU": 55, "replay": 45,
       "CASIA-FASD": 35, "CASIA-SURF": 35, "HQ-WMCA": 50}


def axes_and_C(ds):
    p = os.path.join(ART, f"{ds}_train_resnet50.npz".replace("+", "plus"))
    assert os.path.isfile(p), f"faltan embeddings: {p} (corre 00 primero)"
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float64); y = z["y"].astype(int); st = z["subtype"].astype(str)
    A = all_axes(standardize(X), y, st)
    ks, C = cosine_matrix(A)
    return [str(k) for k in ks], C


def coverage(ks, C, subset):
    idx = [ks.index(k) for k in subset]
    return float(np.mean([max(C[k, j] for j in idx) for k in range(len(ks))]))


def sample_subsets(ks, C, n_target, seed=0):
    """Subconjuntos de tamaños 2..K-1 elegidos para CUBRIR el rango de cobertura.

    Si el nº total de combinaciones es pequeño (K<=5) se toman todas; si es grande, se muestrea
    y se estratifica por cobertura para no concentrar todos los puntos en el mismo sitio."""
    K = len(ks)
    sizes = list(range(2, K)) or [1]
    cand = []
    rng = np.random.default_rng(seed)
    for m in sizes:
        combos = list(itertools.combinations(ks, m))
        if len(combos) > 60:
            pick = rng.choice(len(combos), 60, replace=False)
            combos = [combos[i] for i in pick]
        cand += [list(c) for c in combos]
    covs = np.array([coverage(ks, C, s) for s in cand])
    # estratificar: n_target cuantiles de cobertura, uno por estrato
    order = np.argsort(covs)
    picks, seen = [], set()
    for q in np.linspace(0, len(order) - 1, min(n_target, len(order))):
        i = order[int(round(q))]
        key = tuple(sorted(cand[i]))
        if key in seen: continue
        seen.add(key); picks.append(cand[i])
    return picks


def write_manifest(ds, keep, tag):
    src = os.path.join(MAN_SRC, f"{ds}_subtype.csv")
    dst = os.path.join(MAN_OUT, f"{ds}__{tag}.csv")
    keep = set(keep) | {"live"}
    with open(src) as f, open(dst, "w", newline="") as g:
        rd = csv.DictReader(f); w = csv.writer(g); w.writerow(rd.fieldnames)
        for r in rd:
            # test SIEMPRE intacto. dev: intacto en datasets cuyo protocolo oficial ya separa los
            # tipos de ataque entre splits (CASIA-SURF: train 04/05/06 vs dev/test 01/02/03) — si se
            # filtrara por los subtipos de train, el dev se quedaria SIN ataques y el EER no existe.
            dev_intact = ds in ("CASIA-SURF",)
            if r["split"] == "test" or (r["split"] == "dev" and dev_intact) or r["subtype"] in keep:
                w.writerow([r[c] for c in rd.fieldnames])
    return dst


def main():
    ds = sys.argv[1]
    n_sub = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ks, C = axes_and_C(ds)
    subs = sample_subsets(ks, C, n_sub)
    est = EST.get(ds, 60)
    print(f"== {ds}: {len(ks)} subtipos -> {len(subs)} subconjuntos x {seeds} seeds ==")
    jobs, meta = [], {}
    for i, S in enumerate(subs):
        cov = coverage(ks, C, S)
        tag = f"G{ds.replace('-','').replace('+','p')}c{i:02d}"
        path = write_manifest(ds, S, tag)
        meta[tag] = {"subset": S, "m": len(S), "coverage": cov}
        print(f"  {tag}: m={len(S)} cov={cov:+.3f}  {', '.join(S)}")
        for s in range(seeds):
            jobs.append({
                "block": f"G-cov-{ds}", "label": f"std-{tag}-s{s}", "n_iters": len(S) + 1,
                "est_min": est, "script": "ibt_generic.py",
                "sig": f"--manifest {path} --dataset {ds} --design within --order standard --seed {s} ",
                "args": (f"--manifest {path} --dataset {ds} --design within --order standard "
                         f"--seed {s} --model resnet50 --epochs_per_iter 3 --cap_per_class 4000 "
                         f"--batch_size 64 --tag std-{tag}"),
            })
    mp = os.path.join(ART, "coverage_cells.json")
    allm = json.load(open(mp)) if os.path.isfile(mp) else {}
    allm[ds] = {"ks": ks, "cells": meta}
    json.dump(allm, open(mp, "w"), indent=1)

    jp = os.path.join(ART, "jobs.json")
    cur = json.load(open(jp)); have = {j["label"] for j in cur}
    add = [j for j in jobs if j["label"] not in have]
    cur.extend(add); json.dump(cur, open(jp, "w"), indent=1)
    print(f"  +{len(add)} jobs (~{len(add)*est/60:.0f} GPU-h) -> jobs.json (total {len(cur)})")


if __name__ == "__main__":
    main()
