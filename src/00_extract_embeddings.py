#!/usr/bin/env python3
"""
Embeddings CONGELADOS (ImageNet, sin entrenar en estos datos -> sin leakage) para la
geometría DAXIS. Penúltima capa (avgpool) de resnet50 / resnext101_64x4d.

Corre en el env pytorch (3.8). Salida: artifacts/<dataset>_<split>_<backbone>.npz
  X (n, 2048) float32 · y (n,) int8 · subtype (n,) str · subject (n,) str · path (n,) str

Por CPU por defecto (las GPUs están ocupadas por la campaña; 224px resnet50 CPU ~ ok con cap).
Uso:  python 00_extract_embeddings.py <dataset> <split> <backbone> [cap_per_subtype] [device]
"""
import os, sys, csv
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "fas_benchmark"))
from models.ModelFactory import ModelFactory
from run_experiment import make_tf
from ibt_generic import subject_of

from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa; os.makedirs(ART, exist_ok=True)


def load_rows(dataset, split, cap, seed=0):
    import random
    man = os.path.join(HERE, "..", "fas_benchmark", "manifests", f"{dataset}_subtype.csv")
    by = {}
    for r in csv.DictReader(open(man)):
        if r["dataset"] == dataset and r["split"] == split:
            by.setdefault(r["subtype"], []).append((r["path"], int(r["label"]), r["subtype"]))
    rows = []
    for st, v in sorted(by.items()):
        c = cap * 2 if st == "live" else cap              # mismo cap que el entrenador
        if len(v) > c:
            random.Random(seed).shuffle(v); v = v[:c]
        rows += v
    return rows


@torch.no_grad()
def main():
    ds, split, bb = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    dev = sys.argv[5] if len(sys.argv) > 5 else "cpu"
    out = os.path.join(ART, f"{ds}_{split}_{bb}.npz".replace("+", "plus"))
    if os.path.isfile(out):
        print(f"[skip] ya existe {out}"); return
    torch.set_num_threads(max(8, os.cpu_count() // 2))
    rows = load_rows(ds, split, cap)
    print(f"[{ds}/{split}/{bb}] {len(rows)} imgs (cap {cap}/subtipo) device={dev}", flush=True)
    m = ModelFactory.create(bb, num_classes=2, device=dev, pretrained=True).eval()
    feats = {}
    h = m.model.avgpool.register_forward_hook(lambda mod, i, o: feats.__setitem__("f", o.flatten(1)))
    tf = make_tf(False)
    X, Y, ST, SU, P = [], [], [], [], []
    B = 64; buf, meta = [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(dev); m(x)
        X.append(feats["f"].cpu().numpy().astype(np.float32))
        for pa, yy, st in meta:
            Y.append(yy); ST.append(st); SU.append(subject_of(pa, ds)); P.append(pa)
        buf.clear(); meta.clear()
    done = 0
    for pa, yy, st in rows:
        try:
            buf.append(tf(Image.open(pa).convert("RGB"))); meta.append((pa, yy, st))
        except Exception:
            continue
        if len(buf) >= B:
            flush(); done += B
            if done % 3200 == 0: print(f"  {done}/{len(rows)}", flush=True)
    flush()
    h.remove()
    np.savez_compressed(out, X=np.concatenate(X), y=np.array(Y, dtype=np.int8),
                        subtype=np.array(ST), subject=np.array(SU), path=np.array(P))
    print(f"-> {out} ({len(Y)} filas)", flush=True)


if __name__ == "__main__":
    main()
