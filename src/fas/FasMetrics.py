"""
Métricas FAS para el UniAttackData / Unified Physical-Digital Attack Detection Challenge.

╔══════════════════════════════════════════════════════════════════════════════╗
║  AVISO                                                                          ║
║  `evaluar_modelo` está DEPRECATED y produce métricas que NO coinciden con las   ║
║  del organizador. Usa `evaluar_oficial` (modelo + loaders dev/test) o           ║
║  `evaluar_oficial_desde_fichero` (fichero de predicción). Detalle del fallo y    ║
║  reproducción: ../ablations_revision/DIAGNOSIS.md                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# ──────────────────────────────────────────────────────────────────────────────
# Protocolo oficial (helpers compartidos)
#
# Convención: ATAQUE = clase positiva. `attack_score` = mayor ⇒ más probable ataque.
# Para un modelo con clase 0 = bonafide:  attack_score = 1 - softmax[:, 0].
# Etiqueta binaria: is_attack = 1 si label != 0 (bonafide = label 0).
# El umbral se fija con el EER del split de DESARROLLO (val) y se aplica a TEST.
# ──────────────────────────────────────────────────────────────────────────────
def _eer_threshold(is_attack, attack_score):
    """Umbral tau (predecir ataque si score>=tau) donde APCER==BPCER, y el EER de ese punto."""
    fpr, tpr, thr = roc_curve(is_attack, attack_score, pos_label=1)
    fnr = 1 - tpr                       # APCER: ataques no detectados
    i = np.nanargmin(np.abs(fnr - fpr))  # fpr = BPCER: bonafide marcado como ataque
    return float(thr[i]), float((fnr[i] + fpr[i]) / 2.0)


def _metrics_at(is_attack, attack_score, tau):
    """APCER/BPCER/ACER/ACC con ataque=positivo, prediciendo ataque si score>=tau."""
    pred_attack = (attack_score >= tau).astype(int)
    attack = is_attack == 1
    bona = ~attack
    apcer = float(np.mean(pred_attack[attack] == 0)) if attack.any() else 0.0   # ataque→bonafide
    bpcer = float(np.mean(pred_attack[bona] == 1)) if bona.any() else 0.0        # bonafide→ataque
    return dict(apcer=apcer, bpcer=bpcer, acer=(apcer + bpcer) / 2.0,
                acc=float(np.mean(pred_attack == is_attack)))


def _infer_attack_scores(modelo, dataloader, device="cuda"):
    """Devuelve (is_attack, attack_score) corriendo el modelo. attack_score = 1 - P(clase0)."""
    modelo.to(device)
    modelo.eval()
    scores, labels = [], []
    with torch.no_grad():
        for entradas, etiquetas in dataloader:
            entradas = entradas.to(device)
            probs = F.softmax(modelo(entradas), dim=1)
            scores.append((1.0 - probs[:, 0]).cpu().numpy())   # P(ataque) = 1 - P(bonafide)
            labels.append(np.asarray(etiquetas))
    labels = np.concatenate(labels)
    is_attack = (labels != 0).astype(int)                       # bonafide = clase 0
    return is_attack, np.concatenate(scores)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluador OFICIAL (reproduce al organizador, ~1e-4)
# ──────────────────────────────────────────────────────────────────────────────
def evaluar_oficial(modelo, dev_loader, test_loader, device="cuda", verbose=True):
    """
    Métricas FAS según el protocolo oficial del challenge (ISO/IEC 30107-3):

      1. ATAQUE = clase positiva; score = P(ataque) = 1 - softmax[:, 0].
      2. AUC sobre TEST (todas las familias de ataque juntas, bonafide vs ataque).
      3. Umbral tau = EER calculado en el split de DESARROLLO (dev_loader).
      4. APCER/BPCER/ACER sobre TEST con ese tau fijo.
      5. El EER reportado es el de dev (define tau), no el de test.

    Parámetros:
        modelo:      red entrenada (clase 0 = bonafide).
        dev_loader:  loader de validación (define el umbral).
        test_loader: loader de test (métricas finales).
    Retorna: dict(auc, acer, apcer, bpcer, acc, dev_eer).
    """
    dev_y, dev_s = _infer_attack_scores(modelo, dev_loader, device)
    test_y, test_s = _infer_attack_scores(modelo, test_loader, device)

    tau, dev_eer = _eer_threshold(dev_y, dev_s)
    m = _metrics_at(test_y, test_s, tau)
    auc = float(roc_auc_score(test_y, test_s))
    res = dict(auc=auc, dev_eer=dev_eer, threshold=tau, **m)

    if verbose:
        print(f"[oficial] AUC={auc:.4f} ACER={m['acer']:.4f} "
              f"APCER={m['apcer']:.4f} BPCER={m['bpcer']:.4f} "
              f"ACC={m['acc']:.4f} dev_EER={dev_eer:.4f} (tau={tau:.4f})")
    return res


def evaluar_oficial_desde_fichero(pred_file, val_protocol, test_protocol, verbose=True):
    """
    Igual que `evaluar_oficial` pero a partir de un fichero de predicción
    `<ruta> <score>` que contiene filas de val y de test (formato de submission).

        - score del fichero = se interpreta como score de ataque (mayor ⇒ ataque).
        - val_protocol / test_protocol: ficheros `<ruta> a_b_c` con la verdad.
          a==0 bonafide, a==1 ataque físico, a==2 ataque digital.

    Reproduce los resultados del organizador a ~1e-4. Ver
    ../ablations_revision/fas_official_metrics.py para la versión batch + CSV.
    """
    import os

    def _proto(path):
        gt = {}
        for line in open(path):
            parts = line.strip().split()
            if len(parts) == 2:
                p, label = parts
                split = "val" if "Data-val" in p else "test"
                gt[(split, os.path.basename(p))] = 0 if label.startswith("0") else 1
        return gt

    gt = {}
    gt.update(_proto(val_protocol))
    gt.update(_proto(test_protocol))

    dev, test = [], []
    for line in open(pred_file):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        p, s = parts
        split = "val" if "Data-val" in p else "test"
        key = (split, os.path.basename(p))
        if key in gt:
            (dev if split == "val" else test).append((gt[key], float(s)))

    dev_y = np.array([a for a, _ in dev]);  dev_s = np.array([s for _, s in dev])
    test_y = np.array([a for a, _ in test]); test_s = np.array([s for _, s in test])

    tau, dev_eer = _eer_threshold(dev_y, dev_s)
    m = _metrics_at(test_y, test_s, tau)
    auc = float(roc_auc_score(test_y, test_s))
    res = dict(auc=auc, dev_eer=dev_eer, threshold=tau, **m)
    if verbose:
        print(f"[oficial:{os.path.basename(pred_file)}] AUC={auc:.4f} "
              f"ACER={m['acer']:.4f} ACC={m['acc']:.4f} dev_EER={dev_eer:.4f}")
    return res


# ──────────────────────────────────────────────────────────────────────────────
# DEPRECATED — NO USAR. Mantenida solo para reproducir resultados antiguos.
# ──────────────────────────────────────────────────────────────────────────────
def evaluar_modelo(modelo, dataloader, num_clases, device="cuda", verbose=True, save_file=None):
    """
    ⚠️  DEPRECATED — ESTÁ EQUIVOCADA. NO coincide con la métrica oficial del organizador.

    Usa en su lugar `evaluar_oficial(modelo, dev_loader, test_loader)`.

    Por qué está mal (ver ../ablations_revision/DIAGNOSIS.md, reproducido en 755 ficheros):
      1. UMBRAL IN-SAMPLE: calcula el EER (`best_threshold`) sobre el MISMO conjunto que
         evalúa. El protocolo oficial fija el umbral con el EER del split de DESARROLLO
         (val) y lo aplica a test. → ACER optimista y no comparable. (causa principal)
      2. SUBCONJUNTO: evalúa el dataloader que se le pase (val, a veces filtrado por
         familia), no el TEST completo con todas las familias juntas. → población distinta.
      3. EER REPORTADO: devuelve el EER del propio conjunto evaluado, no el de val.
      4. POLARIDAD/CONVENCIÓN: toma `probs[:, 0]` como "P(bonafide)" y trata bonafide como
         positivo, pero esa columna se comporta como score de ATAQUE (anti-correlada con
         bonafide). → APCER y BPCER quedan intercambiados respecto al significado oficial.

    Se conserva el cuerpo original intacto únicamente para reproducir métricas históricas.
    """
    warnings.warn(
        "evaluar_modelo está DEPRECATED y da métricas incorrectas (umbral in-sample, "
        "subconjunto erróneo, polaridad invertida). Usa evaluar_oficial / "
        "evaluar_oficial_desde_fichero. Ver ablations_revision/DIAGNOSIS.md.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Validación de número de clases
    if num_clases not in [2, 3]:
        raise ValueError("La función soporta solo clasificación binaria o ternaria (2 o 3 clases).")
    modelo.to(device)
    modelo.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    # Iterar sobre el dataloader
    with torch.no_grad():
        for batch in dataloader:
            entradas, etiquetas = batch
            entradas = entradas.to(device)
            etiquetas = etiquetas.to(device)
            # Forward del modelo
            logits = modelo(entradas)
            # Cálculo de pérdida (entropía cruzada, suma sobre el batch)
            loss = F.cross_entropy(logits, etiquetas, reduction='sum')
            total_loss += loss.item()
            total_samples += etiquetas.size(0)
            # Probabilidad de clase 0 (bonafide) usando softmax
            probabilidades = F.softmax(logits, dim=1)
            prob_bonafide = probabilidades[:, 0]  # probabilidad predicha de ser bonafide
            # Almacenar para cálculo global
            all_probs.append(prob_bonafide.cpu().numpy())
            all_labels.append(etiquetas.cpu().numpy())

    # Concatenar todos los resultados
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    # Pérdida promedio en todo el conjunto
    perdida_promedio = total_loss / total_samples

    # Construir etiquetas binarias: 1 = bonafide (clase 0), 0 = ataque (clase != 0)
    y_true = (all_labels == 0).astype(np.int32)

    # # Predicciones binarizadas con umbral 0.5 sobre probabilidad bonafide
    # y_pred = (all_probs >= 0.5).astype(np.int32)

     # Calcular threshold óptimo (mínima diferencia entre FPR y FNR)
    fpr, tpr, thresholds = roc_curve(y_true, all_probs, pos_label=1)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    best_threshold = thresholds[idx_eer]

    print(f"best threshold: {best_threshold}")

    # Aplicar threshold optimizado
    y_pred = (all_probs >= best_threshold).astype(np.int32)

    # Calcular TP, FP, FN, TN
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    # Precisión y recall (evitando división por cero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # AUC (área bajo la curva ROC) respecto a bonafide vs ataque
    auc = roc_auc_score(y_true, all_probs)

    # EER (Equal Error Rate) cálculo mediante la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, all_probs, pos_label=1)
    fnr = 1 - tpr
    # Índice donde |FPR - FNR| es mínimo (aproximación al punto EER)
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2  # valor de EER

    # APCER, BPCER al umbral 0.5, y ACER correspondiente
    apcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # ataques clasificados como bonafide / total ataques
    bpcer = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # bonafide clasificados como ataque / total bonafide
    acer = (apcer + bpcer) / 2.0

    # Imprimir métricas si verbose
    if verbose:
        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Pérdida promedio: {perdida_promedio:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"EER: {eer:.4f}")
        print(f"APCER: {apcer:.4f}")
        print(f"BPCER: {bpcer:.4f}")
        print(f"ACER: {acer:.4f}")

    # Guardar resultados en CSV si se indicó una ruta
    if save_file:
        with open(save_file, "w", newline="") as f:
            # Fila de encabezados
            f.write("Precisión,Recall,Pérdida promedio,AUC,EER,APCER,BPCER,ACER\n")
            # Fila de valores (formato decimal con 4 cifras significativas)
            f.write(f"{precision:.4f},{recall:.4f},{perdida_promedio:.4f},{auc:.4f},"
                    f"{eer:.4f},{apcer:.4f},{bpcer:.4f},{acer:.4f}\n")

    # Devolver un diccionario con las métricas
    return {
        "precision": precision,
        "recall": recall,
        "loss_avg": perdida_promedio,
        "auc": auc,
        "eer": eer,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer
    }
