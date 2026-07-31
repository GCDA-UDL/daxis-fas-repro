#!/usr/bin/env python3
"""
Scores PER-SAMPLE por geometría axial (S1 margen, S2 delta LOO, S3 coherencia) +
VALIDACIÓN con ground truth (CASIA-FASD: 10.359 HR_1 mal etiquetados conocidos) +
DESCUBRIMIENTO (top-k sospechosos para inspección). Corre en /opt/conda/bin/python.

Salida: artifacts/<ds>_scores.csv (path,subtype,y,S1,S2,S3) + artifacts/suspects_<ds>.csv
        + métricas de validación por pantalla.
"""
import os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from daxis_ext import standardize, sample_scores

from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa


def load(ds, bb="resnet50", split="train"):
    p = os.path.join(ART, f"{ds}_{split}_{bb}.npz".replace("+", "plus"))
    z = np.load(p, allow_pickle=True)
    return z["X"].astype(np.float64), z["y"].astype(int), z["subtype"].astype(str), z["path"].astype(str)


def write_scores(ds, split="train"):
    X, y, st, paths = load(ds, split=split)
    S = sample_scores(standardize(X), y, st)
    out = os.path.join(ART, f"{ds}_scores.csv".replace("+", "plus"))
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "subtype", "y", "S1", "S2", "S3"])
        for i in range(len(y)):
            w.writerow([paths[i], st[i], int(y[i]), f"{S['S1'][i]:.5f}", f"{S['S2'][i]:.6f}",
                        "" if np.isnan(S['S3'][i]) else f"{S['S3'][i]:.5f}"])
    # sospechosos: bottom-50 por S1 (margen peor = más "contrario a la media")
    idx = np.argsort(S["S1"])[:50]
    with open(os.path.join(ART, f"suspects_{ds}.csv".replace("+", "plus")), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "S1", "subtype", "y", "path"])
        for r, i in enumerate(idx, 1):
            w.writerow([r, f"{S['S1'][i]:.4f}", st[i], int(y[i]), paths[i]])
    print(f"[{ds}] scores -> {out} · suspects top-50 escritos")
    return out


def validate_casia_fasd():
    """S1 computado BAJO LAS ETIQUETAS VIEJAS (con HR_1 como ataque) debe señalar los HR_1.
    AUROC de (-S1_old) para detectar filas cuya etiqueta vieja != corregida."""
    X, y_corr, st_corr, paths = load("CASIA-FASD")
    old_lab = {}
    bak = os.path.join(HERE, "..", "fas_benchmark", "manifests", "CASIA-FASD.csv.mislabeled_bak")
    for r in csv.DictReader(open(bak)):
        old_lab[r["path"]] = int(r["label"])
    keep = np.array([p in old_lab for p in paths])
    Xk, pk = X[keep], paths[keep]
    y_old = np.array([old_lab[p] for p in pk])
    y_cor = y_corr[keep]
    # subtipo "viejo": HR_1 iba como ataque dentro de la carpeta spoof; el eje se construye
    # por subtipo VIEJO = carpeta (live/spoof por etiqueta vieja); usamos subtipo corregido
    # para los spoof reales y agrupamos los "spoof viejos" por su subtipo corregido (los HR_1
    # mal etiquetados entran al grupo del subtipo que la carpeta les daba: 'warped'? No —
    # la extracción los metía en spoof plano. Grupo = subtipo corregido si ataque real; para
    # HR_1 (real pero marcado ataque) los agrupamos como grupo propio 'HR1MIS' para que
    # contaminen un eje concreto igual que contaminaban el entrenamiento).
    st_old = np.where(y_cor == 1, st_corr[keep], np.where(y_old == 1, "HR1MIS", "live"))
    # ojo: bajo etiquetas viejas, y=1 para HR1MIS
    S = sample_scores(standardize(Xk), y_old, st_old)
    is_mislab = (y_old != y_cor)
    s1 = S["S1"]
    # AUROC de -S1 (margen bajo = sospechoso)
    from sklearn.metrics import roc_auc_score, confusion_matrix
    auroc = roc_auc_score(is_mislab.astype(int), -s1)
    pred = (s1 < 0).astype(int)
    cm = confusion_matrix(is_mislab.astype(int), pred)
    # --- variante GLOBAL: un solo eje live-vs-spoof bajo las etiquetas viejas ---
    # Motivo: si los mal etiquetados forman un GRUPO ENTERO, el eje per-grupo se estima CON ellos
    # -> quedan autoconsistentes e invisibles (punto ciego). El eje global no depende del grupo.
    Xks = standardize(Xk)
    dg = Xks[y_old == 1].mean(0) - Xks[y_old == 0].mean(0)
    dg /= (np.linalg.norm(dg) + 1e-12)
    s1_global = (2 * y_old - 1) * (Xks @ dg)
    auroc_g = roc_auc_score(is_mislab, -s1_global)
    print(f"[CASIA-FASD validación] n={len(pk)} mal_etiquetadas={int(is_mislab.sum())}")
    print(f"  AUROC(-S1 per-grupo) = {auroc:.4f}   <- PUNTO CIEGO: el grupo define su propio eje")
    print(f"  AUROC(-S1 GLOBAL)    = {auroc_g:.4f}   <- detecta el grupo entero mal etiquetado")
    json.dump({"auroc_pergroup": float(auroc), "auroc_global": float(auroc_g),
               "n": int(len(pk)), "n_mislabeled": int(is_mislab.sum()),
               "nota": "per-grupo es ciego al mal-etiquetado sistematico de un grupo completo"},
              open(os.path.join(ART, "validation_HR1.json"), "w"), indent=1)
    print(f"  confusión @S1<0:\n{cm}")
    return auroc


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    if ds in ("HQ-WMCA", "all"): write_scores("HQ-WMCA")
    if ds in ("CelebA-Spoof", "all"):
        try: write_scores("CelebA-Spoof")
        except FileNotFoundError: print("(CelebA aún sin embeddings)")
    if ds in ("CASIA-FASD", "all", "validate"):
        try:
            write_scores("CASIA-FASD")
            validate_casia_fasd()
        except FileNotFoundError: print("(CASIA-FASD aún sin embeddings)")
