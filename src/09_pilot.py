#!/usr/bin/env python3
"""
PROTOCOLO DE PLANIFICACIÓN DE CAPTURA — ¿basta un piloto diminuto para decidir qué PAIs capturar?

Capturar PAIs es caro (máscaras a medida, sesiones de maquillaje, impresión...). Si los ejes
discriminantes estimados con POCAS imágenes por PAI ya eligen el mismo subconjunto que con todos
los datos, se puede planificar la captura: piloto pequeño -> geometría -> decidir qué escalar.

Test (solo CPU, sin entrenar): submuestrear n imgs/PAI, re-estimar ejes, re-ejecutar la selección
por cobertura, y medir el acuerdo con la selección de datos completos (Jaccard) y la pérdida de
cobertura real (evaluada SIEMPRE con la geometría completa, que es la verdad de referencia).

Uso: /opt/conda/bin/python 09_pilot.py [HQ-WMCA]
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
    """Ejes con n_per imgs por PAI (y el mismo nº de bonafide) — simula un piloto."""
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

    print(f"== PILOTO: ¿cuántas imágenes/PAI hacen falta para decidir bien? [{ds}] ==")
    print(f"  selección con datos completos: " + " · ".join(f"m={m}:{cov_full[m]:.3f}" for m in full))
    print(f"\n  {'n/PAI':>6s} {'m':>2s} {'Jaccard vs full':>16s} {'cobertura real':>15s} {'pérdida':>8s}")
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
                cvs.append(cov_true(ks, C_true, S))     # cobertura REAL del set elegido con poco dato
            loss = cov_full[m] - float(np.mean(cvs))
            print(f"  {n_per:6d} {m:2d} {np.mean(js):16.2f} {np.mean(cvs):15.3f} {loss:+8.3f}")
            out["sizes"][str(n_per)][str(m)] = {"jaccard": float(np.mean(js)),
                                                "cov_real": float(np.mean(cvs)), "loss": float(loss)}
        print()
    tot = {k: int(np.sum(st == k)) for k in ks}
    print(f"  (train real por PAI: min={min(tot.values())} max={max(tot.values())})")
    json.dump(out, open(os.path.join(ART, f"pilot_{ds}.json".replace("+", "plus")), "w"), indent=1)
    print(f"  -> artifacts/pilot_{ds}.json")


if __name__ == "__main__":
    main()
