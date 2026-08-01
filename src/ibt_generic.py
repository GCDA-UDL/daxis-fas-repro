#!/usr/bin/env python3
"""
IBT generalizado — curriculum binario-incremental sobre CUALQUIER manifiesto con subtipos.

Motivation: UniAttackData+ has 13 bonafide samples in validation, so honest checkpoint
selection is impossible there and the paired Wilcoxon came out non-significant (p=0.385). Here we use
datasets with a LARGE validation split so the question can actually be decided. The comparison is
always against a MATCHED `standard` baseline (same iterations, epochs and loaders).

Designs:
  --design within  : todos los subtipos en train y test (control; ojo: OULU/SiW saturan).
  --design xtype   : ZERO-SHOT CROSS-TYPE. Los subtipos de --holdout se EXCLUYEN de train/dev y el
                     test se restringe a bonafide + esos subtipos no vistos. Replica el reto real
                     the competition setting, but with a validation split large enough to select on.

Órdenes: freq (frecuente->raro) | reverse | random | standard (all-at-once pareado).
Selection: always on dev (EER); test is only read to report.

Ej:
  python ibt_generic.py --manifest manifests/CelebA-Spoof_subtype.csv --dataset CelebA-Spoof \
      --design xtype --holdout 3DMask,Phone,A4 --order freq --seed 0 --cap_per_class 4000
"""
import os, sys, csv, argparse, random, re
from collections import Counter, defaultdict
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from models.ModelFactory import ModelFactory
from metrics.FasMetrics import evaluar_oficial
from run_experiment import make_tf, class_weights

OUT = os.environ.get("IBT_OUT", "/mnt/d2/ibt26")
RES_DIR = os.path.join(OUT, "results"); CKPT_DIR = os.path.join(OUT, "checkpoints")
CURVE_DIR = os.path.join(RES_DIR, "curves")
for d in (RES_DIR, CKPT_DIR, CURVE_DIR): os.makedirs(d, exist_ok=True)
RES_CSV = os.path.join(RES_DIR, "results_subtype.csv")


def subject_of(path, dataset):
    """Subject identity, so that dev shares no subject with train (prevents leakage)."""
    if dataset == "CelebA-Spoof":
        m = re.search(r"/Data/(?:train|test)/([^/]+)/", path)
        return m.group(1) if m else os.path.basename(os.path.dirname(path))
    if dataset == "CASIA-FASD":                     # .../{live,spoof}/s10v3f0.png -> sujeto s10
        m = re.search(r"/s(\d+)v(?:HR_)?\d+f\d+\.png$", path)
        if m: return f"s{m.group(1)}"
        raise ValueError(f"CASIA-FASD: no puedo extraer sujeto de {path}")
    if dataset == "CASIA-SURF":                     # .../{real,fake}_part/<SUJETO>/x.rssdk/color/f.jpg
        m = re.search(r"_part/([^/]+)/", path)
        if m: return m.group(1)
        raise ValueError(f"CASIA-SURF: no puedo extraer sujeto de {path}")
    d = os.path.basename(os.path.dirname(path))
    if dataset == "OULU-NPU":                      # v_spoof_g1_3_16_4 -> user 16
        p = d.split("_");  return p[4] if len(p) >= 6 else d
    if dataset == "SiW":                            # v_live_g111-2-1-1-2 -> sujeto 111
        m = re.search(r"g(\d+)-", d);  return m.group(1) if m else d
    return d


class ListDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            return self.tf(Image.open(p).convert("RGB")), y
        except Exception:
            # do not mask the problem silently: the caller accounts for it
            return torch.zeros(3, 224, 224), y


def load_manifest(path, dataset):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset:
                rows.append((r["split"], int(r["label"]), r["subtype"], r["path"]))
    return rows


def cap(items, n, seed):
    if not n or len(items) <= n: return items
    rng = random.Random(seed); out = items[:]; rng.shuffle(out); return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--design", default="xtype", choices=["within", "xtype"])
    ap.add_argument("--holdout", default="", help="subtipos no vistos (coma) para design=xtype")
    ap.add_argument("--order", default="freq", choices=["freq", "reverse", "random", "standard"])
    # --- extensiones DAXIS (daxis_experiments): retrocompatibles, without efecto si no se usan ---
    ap.add_argument("--order_list", default=None, help="orden custom de subtipos (coma); ignora --order salvo 'standard'")
    ap.add_argument("--stages_json", default=None, help="JSON: stages=[[ [subtype,frac],... ],...] (rehearsal/merged stages)")
    ap.add_argument("--sample_order", default="shuffle", choices=["shuffle", "easy2hard", "hard2easy"],
                    help="orden intra-etapa por margen axial (requiere --sample_scores)")
    ap.add_argument("--sample_scores", default=None, help="csv con columnas path,S1 (para --sample_order)")
    ap.add_argument("--tag", default=None, help="override del nombre de method: {tag}-s{seed}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs_per_iter", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap_per_class", type=int, default=4000, help="tope de imgs por subtipo (y x2 para bonafide)")
    ap.add_argument("--dev_frac", type=float, default=0.15)
    args = ap.parse_args()
    dev = args.device; E = args.epochs_per_iter
    torch.manual_seed(args.seed); random.seed(args.seed)

    hold = [h for h in args.holdout.split(",") if h]
    method = (f"standard-s{args.seed}" if args.order == "standard" else f"IBT-{args.order}-s{args.seed}")
    if args.tag: method = f"{args.tag}-s{args.seed}"
    tagds = f"{args.dataset}_{args.design}" + (f"_ho-{'+'.join(hold)}" if hold else "")
    tag = f"{tagds}_{args.model}_{method}".replace("+", "plus").replace("/", "_")

    rows = load_manifest(args.manifest, args.dataset)
    assert rows, f"empty manifest for {args.dataset}"

    tr_pool = [r for r in rows if r[0] in ("train",)]
    dv_pool = [r for r in rows if r[0] == "dev"]
    te_pool = [r for r in rows if r[0] == "test"]

    # --- design ---
    if args.design == "xtype":
        assert hold, "design=xtype requiere --holdout"
        tr_pool = [r for r in tr_pool if r[2] == "live" or r[2] not in hold]
        dv_pool = [r for r in dv_pool if r[2] == "live" or r[2] not in hold]
        te_pool = [r for r in te_pool if r[2] == "live" or r[2] in hold]   # test = SOLO no vistos

    # --- dev: use the official split if present, otherwise carve it by subject from train ---
    if not dv_pool:
        subs = sorted({subject_of(r[3], args.dataset) for r in tr_pool})
        rng = random.Random(0); rng.shuffle(subs)
        dev_subs = set(subs[:max(1, int(len(subs) * args.dev_frac))])
        dv_pool = [r for r in tr_pool if subject_of(r[3], args.dataset) in dev_subs]
        tr_pool = [r for r in tr_pool if subject_of(r[3], args.dataset) not in dev_subs]
        print(f"  dev tallado por sujeto: {len(dev_subs)}/{len(subs)} subjects")

    # --- muestreo (tractabilidad) ---
    by_st = defaultdict(list)
    for r in tr_pool: by_st[r[2]].append(r)
    bona = cap([(r[3], 0) for r in by_st.get("live", [])], args.cap_per_class * 2, args.seed)
    spoof_by = {st: cap([(r[3], 1) for r in v], args.cap_per_class, args.seed)
                for st, v in by_st.items() if st != "live"}
    dv = cap([(r[3], r[1]) for r in dv_pool], args.cap_per_class * 4, 0)
    te = cap([(r[3], r[1]) for r in te_pool], args.cap_per_class * 6, 0)

    freq_order = [st for st, _ in Counter({k: len(v) for k, v in spoof_by.items()}).most_common()]
    if args.order == "freq":      order = freq_order
    elif args.order == "reverse": order = freq_order[::-1]
    elif args.order == "random":  order = freq_order[:]; random.Random(args.seed).shuffle(order)
    else:                         order = freq_order[:]
    if args.order_list and args.order != "standard":
        order = [s for s in args.order_list.split(",") if s]
        unknown = set(order) - set(spoof_by)
        assert not unknown, f"--order_list con subtipos desconocidos: {unknown} (hay: {sorted(spoof_by)})"
    # per-sample scores for within-stage ordering
    smap = None
    if args.sample_order != "shuffle":
        assert args.sample_scores, "--sample_order requiere --sample_scores"
        import csv as _csv
        smap = {r["path"]: float(r["S1"]) for r in _csv.DictReader(open(args.sample_scores))}
    print(f"[{tag}] bona={len(bona)} spoof={sum(len(v) for v in spoof_by.values())} "
          f"dev={len(dv)} test={len(te)}")
    print(f"  design={args.design} holdout={hold or '-'} order={'ALL-AT-ONCE' if args.order=='standard' else order}")

    tf_tr, tf_ev = make_tf(True), make_tf(False)
    dl = lambda items, sh, tf: DataLoader(ListDataset(items, tf), batch_size=args.batch_size,
                                          shuffle=sh, num_workers=args.workers, pin_memory=True)
    dev_loader, test_loader = dl(dv, False, tf_ev), dl(te, False, tf_ev)

    model = ModelFactory.create(args.model, num_classes=2, device=dev, pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    curve = open(os.path.join(CURVE_DIR, f"{tag}.csv"), "w", newline=""); cw = csv.writer(curve)
    cw.writerow(["iter", "subtype", "epochs_acum", "train_loss", "train_acc", "test_acer", "test_auc", "dev_eer"])
    curve.flush()
    new = not os.path.isfile(RES_CSV)
    rf = open(RES_CSV, "a", newline=""); rw = csv.writer(rf)
    if new:
        rw.writerow(["dataset", "design", "holdout", "model", "method", "iter", "n_train", "n_test",
                     "auc", "acer", "apcer", "bpcer", "acc", "dev_eer", "train_loss", "train_acc"]); rf.flush()

    if args.stages_json:
        import json as _json
        stages = _json.load(open(args.stages_json))     # [[ [subtipo,frac], ... ], ...]
    else:
        stages = ["__all__"] * (len(order) + 1) if args.order == "standard" else order + ["__consolidation__"]
    ep_acc = 0
    for it, c in enumerate(stages, 1):
        if isinstance(c, list):                          # generalised stage (rehearsal / merged)
            items = list(bona)
            for st, frac in c:
                pool = spoof_by[st]
                n = max(1, int(len(pool) * float(frac)))
                items += pool if n >= len(pool) else random.Random(args.seed * 997 + it).sample(pool, n)
            name = "+".join(st for st, _ in c)[:48]
        elif c in ("__consolidation__", "__all__"):
            items = bona + [x for v in spoof_by.values() for x in v]; name = "ALL"
        else:
            items = bona + spoof_by[c]; name = c
        if smap is not None:                             # orden intra-etapa por margen axial
            sgn = -1 if args.sample_order == "easy2hard" else 1   # easy2hard = margen desc
            items = sorted(items, key=lambda pt: sgn * smap.get(pt[0], 0.0))
        loader = dl(items, smap is None, tf_tr)          # con orden fijado: shuffle=False
        crit = nn.CrossEntropyLoss(weight=class_weights([y for _, y in items], dev))
        print(f"  iter {it}/{len(stages)}  bona+{name}  n={len(items)}")
        for _ in range(E):
            model.train(); tot = correct = 0; lsum = 0.0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); o = model(x); loss = crit(o, y); loss.backward(); opt.step()
                lsum += loss.item() * y.size(0); tot += y.size(0); correct += (o.argmax(1) == y).sum().item()
            ep_acc += 1
        tl, ta = lsum / tot, correct / tot
        res = evaluar_oficial(model, dev_loader, test_loader, device=dev, verbose=False)
        cw.writerow([it, name, ep_acc, f"{tl:.4f}", f"{ta:.4f}", f"{res['acer']:.4f}",
                     f"{res['auc']:.4f}", f"{res['dev_eer']:.4f}"]); curve.flush()
        torch.save({"state_dict": model.state_dict(), "iter": it, "subtype": name, "metrics": res},
                   os.path.join(CKPT_DIR, f"{tag}_iter{it}.pth"))
        rw.writerow([args.dataset, args.design, "+".join(hold), args.model, method, it, len(items), len(te),
                     f"{res['auc']:.4f}", f"{res['acer']:.4f}", f"{res['apcer']:.4f}", f"{res['bpcer']:.4f}",
                     f"{res['acc']:.4f}", f"{res['dev_eer']:.4f}", f"{tl:.4f}", f"{ta:.4f}"]); rf.flush()
        print(f"    -> ACER={res['acer']:.4f} AUC={res['auc']:.4f} dev_EER={res['dev_eer']:.4f}")
    curve.close(); rf.close()
    print(f"DONE {tag}: {len(stages)} iters, {ep_acc} epochs")


if __name__ == "__main__":
    main()
