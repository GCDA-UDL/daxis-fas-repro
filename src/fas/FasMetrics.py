"""
FAS metrics following the official challenge protocol (ISO/IEC 30107-3).

WARNING: `evaluar_modelo` is DEPRECATED and produces metrics that do NOT match the
challenge organisers'. Use `evaluar_oficial` (model plus dev/test loaders) or
`evaluar_oficial_desde_fichero` (a prediction file) instead. The function names are kept in
Spanish because published results reference them; the behaviour is documented here in English.
"""
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# ──────────────────────────────────────────────────────────────────────────────
# Official protocol (shared helpers)
#
# Convention: attack is the positive class. A higher `attack_score` means more likely an attack.
# For a model whose class 0 is bonafide: attack_score = 1 - softmax[:, 0].
# label binaria: is_attack = 1 si label != 0 (bonafide = label 0).
# The threshold is set at the EER of the DEVELOPMENT split and applied to TEST.
# ──────────────────────────────────────────────────────────────────────────────
def _eer_threshold(is_attack, attack_score):
    """Threshold tau (predict attack if score>=tau) where APCER==BPCER, and the EER at that point."""
    fpr, tpr, thr = roc_curve(is_attack, attack_score, pos_label=1)
    fnr = 1 - tpr                       # APCER: ataques no detectados
    i = np.nanargmin(np.abs(fnr - fpr))  # fpr = BPCER: bonafide marcado como ataque
    return float(thr[i]), float((fnr[i] + fpr[i]) / 2.0)


def _metrics_at(is_attack, attack_score, tau):
    """APCER/BPCER/ACER/ACC with attack as positive, predicting attack if score>=tau."""
    pred_attack = (attack_score >= tau).astype(int)
    attack = is_attack == 1
    bona = ~attack
    apcer = float(np.mean(pred_attack[attack] == 0)) if attack.any() else 0.0   # ataque→bonafide
    bpcer = float(np.mean(pred_attack[bona] == 1)) if bona.any() else 0.0        # bonafide→ataque
    return dict(apcer=apcer, bpcer=bpcer, acer=(apcer + bpcer) / 2.0,
                acc=float(np.mean(pred_attack == is_attack)))


def _infer_attack_scores(modelo, dataloader, device="cuda"):
    """Returns (is_attack, attack_score) by running the model. attack_score = 1 - P(class 0)."""
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
# Official evaluator (reproduces the organisers' numbers to ~1e-4)
# ──────────────────────────────────────────────────────────────────────────────
def evaluar_oficial(modelo, dev_loader, test_loader, device="cuda", verbose=True):
    """
    FAS metrics under the official challenge protocol (ISO/IEC 30107-3):

      1. ATTACK is the positive class; score = P(attack) = 1 - softmax[:, 0].
      2. AUC over TEST (all attack families pooled, bonafide against attack).
      3. Threshold tau = the EER computed on the DEVELOPMENT split (dev_loader).
      4. APCER/BPCER/ACER over TEST at that fixed tau.
      5. The reported EER is the dev one (which defines tau), not the test EER.

    Args:
        modelo:      trained network (class 0 = bonafide).
        dev_loader:  validation loader (sets the threshold).
        test_loader: test loader (final metrics).
    Returns: dict(auc, acer, apcer, bpcer, acc, dev_eer).
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
    Same as `evaluar_oficial` but starting from a prediction file rather than a model.

    The label code a_b_c encodes: a==0 bonafide, a==1 physical attack, a==2 digital attack.
    See the batch/CSV variant in the analysis scripts.
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
# DEPRECATED, DO NOT USE. Kept only to reproduce older results.
# ──────────────────────────────────────────────────────────────────────────────
def evaluar_modelo(modelo, dataloader, num_clases, device="cuda", verbose=True, save_file=None):
    """
    DEPRECATED and INCORRECT. It does not match the organisers' official metric.

    Why it is wrong (reproduced over 755 files):
      1. THRESHOLD: it picks the threshold on the same split it evaluates. The official
         protocol fixes the threshold at the EER of the DEVELOPMENT split and applies it to test.
      2. SUBSET: it evaluates whatever dataloader it is given (validation, sometimes filtered by
         attack family) rather than the full test set with all families pooled, so the
         population differs.

    Kept only so that previously published numbers can be reproduced.

    """
    warnings.warn(
        "evaluar_modelo is DEPRECATED and returns incorrect metrics (in-sample threshold, "
        "wrong subset, inverted polarity). Use evaluar_oficial / "
        "evaluar_oficial_desde_fichero. Ver ablations_revision/DIAGNOSIS.md.",
        DeprecationWarning,
        stacklevel=2,
    )

    # validate the number of classes
    if num_clases not in [2, 3]:
        raise ValueError("This function supports only binary or ternary classification (2 or 3 classes).")
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
            # loss (cross-entropy, summed over the batch)
            loss = F.cross_entropy(logits, etiquetas, reduction='sum')
            total_loss += loss.item()
            total_samples += etiquetas.size(0)
            # Probabilidad de clase 0 (bonafide) usando softmax
            probabilidades = F.softmax(logits, dim=1)
            prob_bonafide = probabilidades[:, 0]  # probabilidad predicha de ser bonafide
            # accumulate for the global computation
            all_probs.append(prob_bonafide.cpu().numpy())
            all_labels.append(etiquetas.cpu().numpy())

    # Concatenar todos los resultados
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    # average loss over the whole set
    perdida_promedio = total_loss / total_samples

    # Construir labels binarias: 1 = bonafide (clase 0), 0 = attack (clase != 0)
    y_true = (all_labels == 0).astype(np.int32)

    # # Predicciones binarizadas con threshold 0.5 sobre probabilidad bonafide
    # y_pred = (all_probs >= 0.5).astype(np.int32)

     # optimal threshold (minimum |FPR - FNR|)
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

    # precision and recall (avoiding division by zero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # AUC (area under the ROC) for bonafide vs attack
    auc = roc_auc_score(y_true, all_probs)

    # EER computed from the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, all_probs, pos_label=1)
    fnr = 1 - tpr
    # index where |FPR - FNR| is minimal (approximates the EER point)
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2  # valor de EER

    # APCER, BPCER al threshold 0.5, y ACER correspondiente
    apcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # ataques clasificados como bonafide / total ataques
    bpcer = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # bonafide clasificados como ataque / total bonafide
    acer = (apcer + bpcer) / 2.0

    # print metrics when verbose
    if verbose:
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Average loss: {perdida_promedio:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"EER: {eer:.4f}")
        print(f"APCER: {apcer:.4f}")
        print(f"BPCER: {bpcer:.4f}")
        print(f"ACER: {acer:.4f}")

    # write results to CSV if a path was given
    if save_file:
        with open(save_file, "w", newline="") as f:
            # Fila de encabezados
            f.write("Precision,Recall,AverageLoss,AUC,EER,APCER,BPCER,ACER\n")
            # Fila de valores (formato decimal con 4 cifras significativas)
            f.write(f"{precision:.4f},{recall:.4f},{perdida_promedio:.4f},{auc:.4f},"
                    f"{eer:.4f},{apcer:.4f},{bpcer:.4f},{acer:.4f}\n")

    # return the metrics as a dict
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
