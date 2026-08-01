#!/usr/bin/env python3
"""
Harness FAS v2 — entrena hasta N épocas y, en cada HITO {5,10,20,30,40,50}, guarda:
  - checkpoint (state_dict)            -> /mnt/d2/ibt26/checkpoints/<ds>_<model>_ep<N>.pth
  - evaluación OFICIAL (ACER/AUC/EER)  -> /mnt/d2/ibt26/results/results.csv  (1 fila por hito)
  - curva por época (train loss/acc)   -> /mnt/d2/ibt26/results/curves/<ds>_<model>.csv
Usa class weights balanceados (arregla el desbalance, p.ej. UniAttackData+ 1:26).

Ejemplos:
  python run_experiment.py --dataset UniAttackData+ --model resnet50 --epochs 50
  python run_experiment.py --dataset CASIA-FASD --model resnet50 --epochs 5 --limit 2000   # smoke
"""
import os, sys, csv, argparse, random, re
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from models.ModelFactory import ModelFactory
from metrics.FasMetrics import evaluar_oficial

OUT = os.environ.get("IBT_OUT", "/mnt/d2/ibt26")
RES_DIR = os.path.join(OUT, "results"); CKPT_DIR = os.path.join(OUT, "checkpoints")
CURVE_DIR = os.path.join(RES_DIR, "curves")
for d in (RES_DIR, CKPT_DIR, CURVE_DIR): os.makedirs(d, exist_ok=True)

IMAGENET = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
def make_tf(train, size=224):
    aug = [transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([transforms.Resize((size,size)), *aug, transforms.ToTensor(), IMAGENET])

class ManifestDataset(Dataset):
    def __init__(self, manifest, dataset, split, transform, limit=None):
        self.rows = []
        with open(manifest) as f:
            for r in csv.DictReader(f):
                if r["dataset"] == dataset and r["split"] == split:
                    self.rows.append((r["path"], int(r["label"])))
        if limit:
            random.Random(0).shuffle(self.rows); self.rows = self.rows[:limit]
        self.transform = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        # robusto a images corruptas/ilegibles: reintenta con otras al azar, no rompe el run
        for k in range(20):
            idx = i if k == 0 else random.randrange(len(self.rows))
            p, y = self.rows[idx]
            try:
                return self.transform(Image.open(p).convert("RGB")), y
            except Exception:
                continue
        return torch.zeros(3, 224, 224), self.rows[i][1]

def subject_of(path, dataset):
    base = os.path.basename(path); parent = os.path.basename(os.path.dirname(path))
    if dataset == "CASIA-FASD":
        m = re.search(r"s(\d+)", base);   return f"s{m.group(1)}" if m else base
    if dataset == "SiW":
        m = re.search(r"g(\d+)", parent); return f"g{m.group(1)}" if m else parent
    if dataset == "OULU-NPU":
        m = re.search(r"g\d+_\d+_(\d+)_\d+", parent); return m.group(1) if m else parent
    if dataset == "CelebA-Spoof":      # .../Data/train/<id>/live|spoof/x.jpg -> id
        m = re.search(r"/Data/(?:train|test)/([^/]+)/", path); return m.group(1) if m else parent
    return parent

def split_train_dev_by_subject(rows, dataset, frac=0.15, seed=0):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[subject_of(r[0], dataset)].append(i)
    subs = list(groups); random.Random(seed).shuffle(subs)
    target, dev = int(len(rows)*frac), set()
    for s in subs:
        if len(dev) >= target: break
        dev.update(groups[s])
    dev_idx = sorted(dev); tr_idx = [i for i in range(len(rows)) if i not in dev]
    print(f"  dev por sujeto: {len(subs)} subjects -> dev {len(dev_idx)} muestras (disjuntos de train)")
    return tr_idx, dev_idx

def class_weights(labels, device):
    c = Counter(labels); n = len(labels); k = len(c)
    w = torch.tensor([n/(k*c[i]) for i in range(k)], dtype=torch.float32, device=device)
    print(f"  class weights: {dict(c)} -> {w.tolist()}")
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs", type=int, default=50, help="máximo de épocas")
    ap.add_argument("--milestones", default="5,10,20,30,40,50", help="épocas donde guardar ckpt+eval")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_weights", action="store_true", help="desactivar class weights")
    args = ap.parse_args()
    dev = args.device
    milestones = sorted(int(x) for x in args.milestones.split(","))
    max_ep = max(args.epochs, max(milestones))
    tag = f"{args.dataset}_{args.model}".replace("+","plus").replace("/","_")

    man = os.path.join(HERE, "manifests", f"{args.dataset}.csv")
    assert os.path.isfile(man), f"no existe {man}"
    tf_tr, tf_ev = make_tf(True), make_tf(False)
    train_full = ManifestDataset(man, args.dataset, "train", tf_tr, args.limit)
    has_dev = any(r["split"]=="dev" for r in csv.DictReader(open(man)))
    if has_dev:
        dev_ds = ManifestDataset(man, args.dataset, "dev", tf_ev, args.limit); train_ds = train_full
    else:
        tr_idx, dv_idx = split_train_dev_by_subject(train_full.rows, args.dataset, args.dev_frac)
        dev_rows = [train_full.rows[i] for i in dv_idx]
        train_full.rows = [train_full.rows[i] for i in tr_idx]
        dev_ds = ManifestDataset.__new__(ManifestDataset); dev_ds.rows = dev_rows; dev_ds.transform = tf_ev
        train_ds = train_full
    test_ds = ManifestDataset(man, args.dataset, "test", tf_ev, args.limit)
    print(f"[{args.dataset}/{args.model}] train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)} "
          f"max_ep={max_ep} hitos={milestones}")

    dl = lambda d,s: DataLoader(d, batch_size=args.batch_size, shuffle=s, num_workers=args.workers, pin_memory=True)
    train_loader, dev_loader, test_loader = dl(train_ds,True), dl(dev_ds,False), dl(test_ds,False)

    model = ModelFactory.create(args.model, num_classes=2, device=dev, pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    w = None if args.no_weights else class_weights([y for _,y in train_ds.rows], dev)
    crit = nn.CrossEntropyLoss(weight=w)

    curve_path = os.path.join(CURVE_DIR, f"{tag}.csv")
    cf = open(curve_path, "w", newline=""); cw = csv.writer(cf)
    cw.writerow(["epoch","train_loss","train_acc","val_acer","val_auc","test_acer","test_auc","dev_eer"]); cf.flush()
    res_path = os.path.join(RES_DIR, "results.csv"); res_new = not os.path.isfile(res_path)
    rf = open(res_path, "a", newline=""); rw = csv.writer(rf)
    if res_new:
        rw.writerow(["dataset","model","method","epoch","n_train","n_test","auc","acer","apcer","bpcer","acc","dev_eer","train_loss","train_acc"]); rf.flush()

    for ep in range(1, max_ep+1):
        model.train(); tot=correct=0; lsum=0.0
        for x,y in train_loader:
            x,y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = model(x); loss = crit(out,y); loss.backward(); opt.step()
            lsum += loss.item()*y.size(0); tot += y.size(0); correct += (out.argmax(1)==y).sum().item()
        tl, ta = lsum/tot, correct/tot
        row = [ep, f"{tl:.4f}", f"{ta:.4f}", "", "", "", "", ""]
        if ep in milestones:
            res = evaluar_oficial(model, dev_loader, test_loader, device=dev, verbose=False)
            row[3:8] = [f"{res['acer']:.4f}", f"{res['auc']:.4f}", f"{res['acer']:.4f}", f"{res['auc']:.4f}", f"{res['dev_eer']:.4f}"]
            ckpt = os.path.join(CKPT_DIR, f"{tag}_ep{ep}.pth")
            torch.save({"state_dict": model.state_dict(), "dataset": args.dataset, "model": args.model,
                        "epoch": ep, "metrics": res}, ckpt)
            rw.writerow([args.dataset,args.model,"standard",ep,len(train_ds),len(test_ds),
                         f"{res['auc']:.4f}",f"{res['acer']:.4f}",f"{res['apcer']:.4f}",f"{res['bpcer']:.4f}",
                         f"{res['acc']:.4f}",f"{res['dev_eer']:.4f}",f"{tl:.4f}",f"{ta:.4f}"]); rf.flush()
            print(f"  epoch {ep}: loss={tl:.4f} acc={ta:.4f} | ACER={res['acer']:.4f} AUC={res['auc']:.4f} -> ckpt+eval guardados")
        else:
            print(f"  epoch {ep}: loss={tl:.4f} acc={ta:.4f}")
        cw.writerow(row); cf.flush()
    cf.close(); rf.close()
    print(f"DONE {tag}: curvas={curve_path}")

if __name__ == "__main__":
    main()
