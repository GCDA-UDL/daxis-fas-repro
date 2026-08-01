#!/usr/bin/env python3
"""
Derived manifests: leave-out of classes and per-sample filtering.

Writes filtered copies of a subtype manifest so that training a variant needs no change to the
trainer - only a different manifest.

Usage: python 03_derived_manifests.py [dataset]
"""
import os, sys, csv, json, random

HERE = os.path.dirname(os.path.abspath(__file__))
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa; MAN = os.path.join(HERE, "manifests")
os.makedirs(MAN, exist_ok=True)
DS = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
BASE = os.path.join(HERE, "..", "fas_benchmark", "manifests", f"{DS}_subtype.csv")

rows = list(csv.DictReader(open(BASE)))
HDR = ["dataset", "split", "label", "subtype", "path"]


def write(name, keep_fn):
    out = os.path.join(MAN, f"{DS}__{name}.csv".replace("+", "plus"))
    n_in = n_out = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR)
        for r in rows:
            n_in += 1
            if keep_fn(r):
                w.writerow([r[h] for h in HDR]); n_out += 1
    print(f"  {name}: {n_out}/{n_in} rows -> {os.path.basename(out)}")
    return out


def main():
    O = json.load(open(os.path.join(ART, "orders.json")))[DS]
    picks = O["picks"]
    sco = {}
    sp = os.path.join(ART, f"{DS}_scores.csv".replace("+", "plus"))
    for r in csv.DictReader(open(sp)):
        sco[r["path"]] = float(r["S1"])

    # --- leave-out (solo train; spoof del subtype excluido fuera, live intacto) ---
    def drop_subtypes(drops):
        ds_ = set(drops)
        return lambda r: not (r["split"] == "train" and r["label"] == "1" and r["subtype"] in ds_)
    write(f"L1_drop_{picks['L1_redundant_drop']}", drop_subtypes([picks["L1_redundant_drop"]]))
    write(f"L2_drop_{picks['L2_outlier_drop']}", drop_subtypes([picks["L2_outlier_drop"]]))
    all_sts = sorted({r["subtype"] for r in rows if r["label"] == "1" and r["split"] == "train"})
    for m, keep in O["picks"]["L3_topm_aligned"].items():
        write(f"L3_top{m}aligned", drop_subtypes([s for s in all_sts if s not in keep]))
    for m, keep in O["picks"]["L3b_diverse"].items():
        write(f"L3b_top{m}diverse", drop_subtypes([s for s in all_sts if s not in keep]))
    for m in ("2", "4", "6", "8"):
        for sd in (0, 1, 2):
            keep = random.Random(100 + sd).sample(all_sts, int(m))
            write(f"L3r_top{m}rand{sd}", drop_subtypes([s for s in all_sts if s not in keep]))

    # --- filtrado per-sample (solo train) ---
    tr_paths = [r["path"] for r in rows if r["split"] == "train" and r["path"] in sco]
    ranked = sorted(tr_paths, key=lambda p: sco[p])          # asc: peor margen primero
    n = len(ranked)
    for q in (5, 20):
        k = int(n * q / 100)
        bot = set(ranked[:k]); top = set(ranked[-k:])
        write(f"T1_dropS1bot{q}", lambda r, b=bot: r["path"] not in b or r["split"] != "train")
        write(f"T1_dropS1top{q}", lambda r, t=top: r["path"] not in t or r["split"] != "train")
        for sd in (0, 1, 2):
            rnd = set(random.Random(200 + sd).sample(tr_paths, k))
            write(f"T1_droprand{q}_{sd}", lambda r, x=rnd: r["path"] not in x or r["split"] != "train")

    # --- val-clean (dev only; used by the retro analysis, not for training) ---
    # dev scores do not exist (step 02 covers train only); they are computed on the fly later
    # generamos la variante estructural si hay scores de dev en el csv (opcional).
    print("done.")


if __name__ == "__main__":
    main()
