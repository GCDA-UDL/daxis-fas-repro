#!/usr/bin/env python3
"""
O8: geometry-ADAPTIVE curriculum. After each stage the axes are recomputed
discriminantes con el MODELO ACTUAL (no el backbone congelado) y se elige el siguiente
and the next PAI is the one with the highest cosine to the aggregate of what is trained.

Question: must the geometry follow the model, or is the frozen one enough (O3)?
Mismo protocolo/salidas que ibt_generic (results_subtype.csv, method=IBT-daxisO8ad-sN).
"""
import os, sys, csv, argparse, random
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "fas_benchmark"))
from models.ModelFactory import ModelFactory
from metrics.FasMetrics import evaluar_oficial
from run_experiment import make_tf, class_weights
from ibt_generic import ListDataset, load_manifest, cap, subject_of
from daxis_ext import standardize, all_axes, aggregate_axis

OUT = os.environ.get("IBT_OUT", "/mnt/d2/ibt26_daxis")
RES_DIR = os.path.join(OUT, "results"); CKPT_DIR = os.path.join(OUT, "checkpoints")
CURVE_DIR = os.path.join(RES_DIR, "curves")
for d in (RES_DIR, CKPT_DIR, CURVE_DIR): os.makedirs(d, exist_ok=True)
RES_CSV = os.path.join(RES_DIR, "results_subtype.csv")


@torch.no_grad()
def embed(model, items, tf, device, bs=64, limit=600, seed=0):
    """Penultimate-layer (avgpool) embeddings of the CURRENT model over a subsample."""
    it = items if len(items) <= limit else random.Random(seed).sample(items, limit)
    feats = {}
    h = model.model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o.flatten(1)))
    X = []
    model.eval()
    for i in range(0, len(it), bs):
        from PIL import Image
        xs = []
        for p, _ in it[i:i + bs]:
            try: xs.append(tf(Image.open(p).convert("RGB")))
            except Exception: pass
        if not xs: continue
        model(torch.stack(xs).to(device))
        X.append(feats["f"].cpu().numpy())
    h.remove(); model.train()
    return np.concatenate(X) if X else np.zeros((0, 2048))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--design", default="within", choices=["within", "xtype"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs_per_iter", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap_per_class", type=int, default=4000)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--tag", default="IBT-daxisO8ad")
    args = ap.parse_args()
    dev = args.device; E = args.epochs_per_iter
    torch.manual_seed(args.seed); random.seed(args.seed)
    method = f"{args.tag}-s{args.seed}"
    tag = f"{args.dataset}_{args.design}_{args.model}_{method}".replace("+", "plus")

    rows = load_manifest(args.manifest, args.dataset)
    hold = [h for h in args.holdout.split(",") if h]
    tr_pool = [r for r in rows if r[0] == "train"]
    dv_pool = [r for r in rows if r[0] == "dev"]
    te_pool = [r for r in rows if r[0] == "test"]
    if args.design == "xtype":
        tr_pool = [r for r in tr_pool if r[2] == "live" or r[2] not in hold]
        dv_pool = [r for r in dv_pool if r[2] == "live" or r[2] not in hold]
        te_pool = [r for r in te_pool if r[2] == "live" or r[2] in hold]
    if not dv_pool:
        subs = sorted({subject_of(r[3], args.dataset) for r in tr_pool})
        rng = random.Random(0); rng.shuffle(subs)
        dsubs = set(subs[:max(1, int(len(subs) * args.dev_frac))])
        dv_pool = [r for r in tr_pool if subject_of(r[3], args.dataset) in dsubs]
        tr_pool = [r for r in tr_pool if subject_of(r[3], args.dataset) not in dsubs]

    from collections import defaultdict, Counter
    by = defaultdict(list)
    for r in tr_pool: by[r[2]].append(r)
    bona = cap([(r[3], 0) for r in by.get("live", [])], args.cap_per_class * 2, args.seed)
    spoof_by = {st: cap([(r[3], 1) for r in v], args.cap_per_class, args.seed)
                for st, v in by.items() if st != "live"}
    dv = cap([(r[3], r[1]) for r in dv_pool], args.cap_per_class * 4, 0)
    te = cap([(r[3], r[1]) for r in te_pool], args.cap_per_class * 6, 0)
    start = max(spoof_by, key=lambda k: len(spoof_by[k]))
    print(f"[{tag}] bona={len(bona)} spoof={sum(map(len, spoof_by.values()))} dev={len(dv)} test={len(te)} start={start}")

    tf_tr, tf_ev = make_tf(True), make_tf(False)
    from torch.utils.data import DataLoader
    dl = lambda items, sh, tf: DataLoader(ListDataset(items, tf), batch_size=args.batch_size,
                                          shuffle=sh, num_workers=args.workers, pin_memory=True)
    dev_loader, test_loader = dl(dv, False, tf_ev), dl(te, False, tf_ev)
    model = ModelFactory.create(args.model, num_classes=2, device=dev, pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    curve = open(os.path.join(CURVE_DIR, f"{tag}.csv"), "w", newline=""); cw = csv.writer(curve)
    cw.writerow(["iter", "subtype", "epochs_acum", "train_loss", "train_acc", "test_acer", "test_auc", "dev_eer"]); curve.flush()
    new = not os.path.isfile(RES_CSV)
    rf = open(RES_CSV, "a", newline=""); rw = csv.writer(rf)
    if new:
        rw.writerow(["dataset", "design", "holdout", "model", "method", "iter", "n_train", "n_test",
                     "auc", "acer", "apcer", "bpcer", "acc", "dev_eer", "train_loss", "train_acc"]); rf.flush()

    trained, left = [start], [k for k in spoof_by if k != start]
    ep_acc = 0; it = 0
    while True:
        it += 1
        cur = trained[-1] if it <= len(trained) else None
        if it <= len(trained):
            items = bona + spoof_by[trained[it - 1]]; name = trained[it - 1]
        else:
            items = bona + [x for v in spoof_by.values() for x in v]; name = "ALL"
        loader = dl(items, True, tf_tr)
        crit = nn.CrossEntropyLoss(weight=class_weights([y for _, y in items], dev))
        print(f"  iter {it} bona+{name} n={len(items)}")
        for _ in range(E):
            model.train(); tot = corr = 0; ls = 0.0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); o = model(x); l = crit(o, y); l.backward(); opt.step()
                ls += l.item() * y.size(0); tot += y.size(0); corr += (o.argmax(1) == y).sum().item()
            ep_acc += 1
        tl, ta = ls / tot, corr / tot
        res = evaluar_oficial(model, dev_loader, test_loader, device=dev, verbose=False)
        cw.writerow([it, name, ep_acc, f"{tl:.4f}", f"{ta:.4f}", f"{res['acer']:.4f}", f"{res['auc']:.4f}",
                     f"{res['dev_eer']:.4f}"]); curve.flush()
        torch.save({"state_dict": model.state_dict(), "iter": it, "subtype": name, "metrics": res},
                   os.path.join(CKPT_DIR, f"{tag}_iter{it}.pth"))
        rw.writerow([args.dataset, args.design, "+".join(hold), args.model, method, it, len(items), len(te),
                     f"{res['auc']:.4f}", f"{res['acer']:.4f}", f"{res['apcer']:.4f}", f"{res['bpcer']:.4f}",
                     f"{res['acc']:.4f}", f"{res['dev_eer']:.4f}", f"{tl:.4f}", f"{ta:.4f}"]); rf.flush()
        print(f"    -> ACER={res['acer']:.4f} AUC={res['auc']:.4f} dev_EER={res['dev_eer']:.4f}")
        if name == "ALL": break
        if left:
            # ADAPTATIVO: axes con el modelo ACTUAL sobre subsamples de live + candidatos + entrenados
            groups = {"live": bona}
            groups.update({k: spoof_by[k] for k in set(left) | set(trained)})
            Xs, ys, sts = [], [], []
            for g, its_ in groups.items():
                Z = embed(model, its_, tf_ev, dev, limit=400, seed=args.seed)
                Xs.append(Z); ys += [0 if g == "live" else 1] * len(Z); sts += [g] * len(Z)
            Xc = standardize(np.concatenate(Xs)); ya = np.array(ys); sa = np.array(sts)
            axes = all_axes(Xc, ya, sa)
            agg = aggregate_axis(Xc, ya, sa, trained)
            cos = {k: float(axes[k] @ agg) for k in left}
            nxt = max(cos, key=cos.get)
            print(f"    [adaptivo] cos={ {k: round(v,3) for k,v in sorted(cos.items(), key=lambda kv:-kv[1])} } -> {nxt}")
            trained.append(nxt); left.remove(nxt)
        # once left empties, the next pass is the consolidation stage
    curve.close(); rf.close()
    print(f"DONE {tag}: {it} iters, {ep_acc} epochs · final order: {trained}")


if __name__ == "__main__":
    main()
