#!/usr/bin/env python3
"""
SELECTION AT AN EQUAL IMAGE BUDGET, plus weighted coverage variants.

Fixes a confound in the earlier blocks: comparing strategies at an equal NUMBER OF CLASSES is not
fair, because each subset carries a different amount of data. On HQ-WMCA at m=4 the aligned pick
brought 11,832 images and the coverage pick 6,144 (it chose Wig=624 and Tattoo=576, the smallest
classes). Coverage selection lost by 2.3 AUC while holding MORE coverage - the data deficit ate
the angular advantage. Here the IMAGE budget is fixed and only its allocation varies.

Strategies (all spend the same number of attack images):
  cov      pure coverage (greedy facility location)
  cb       cost-benefit: greedy on  dCoverage / cost   <- the classic submodular knapsack form
  wcov     coverage WEIGHTED by effective size: dCov * log(1+n_k)
  big      the largest classes (control: data volume only)
  rand     random classes (control)
Allocation within a subset: water-filling (as even as availability allows).

Usage: python 13_budget_select.py [HQ-WMCA]
"""
import os, sys, csv, json, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
MAN_OUT = os.path.join(HERE, "manifests")
MAN_SRC = os.path.join(HERE, "..", "fas_benchmark", "manifests")
os.makedirs(MAN_OUT, exist_ok=True)
BUDGETS = [4000, 8000, 16000]        # total ATTACK images
SEEDS = 3


def load_axes(ds):
    z = np.load(os.path.join(ART, f"axes_{ds}.npz".replace("+", "plus")), allow_pickle=True)
    return [str(k) for k in z["ks"]], z["C"]


def counts(ds):
    c = {}
    for r in csv.DictReader(open(os.path.join(MAN_SRC, f"{ds}_subtype.csv"))):
        if r["split"] == "train" and r["subtype"] != "live":
            c[r["subtype"]] = c.get(r["subtype"], 0) + 1
    return c


def cov_of(ks, C, S):
    if not S: return -1.0
    idx = [ks.index(k) for k in S]
    return float(np.mean([max(C[k, j] for j in idx) for k in range(len(ks))]))


def waterfill(sel, cnt, budget):
    """Spreads `budget` images across `sel` as evenly as possible, without exceeding what
    disponible en cada clase. Devuelve {clase: n} o None si entre all no llegan al presupuesto."""
    if not sel: return None
    if sum(cnt[k] for k in sel) < budget: return None
    alloc = {k: 0 for k in sel}
    remaining = budget
    active = list(sel)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for k in list(active):
            room = cnt[k] - alloc[k]
            give = min(share, room, remaining)
            if give > 0:
                alloc[k] += give; remaining -= give; progressed = True
            if alloc[k] >= cnt[k]: active.remove(k)
            if remaining == 0: break
        if not progressed: break
    return alloc if remaining == 0 else None


def pick(strategy, ks, C, cnt, budget, seed=0):
    """Picks classes until the budget can be covered; returns the per-class allocation."""
    avail = [k for k in ks if cnt.get(k, 0) > 0]
    if strategy == "rand":
        rng = random.Random(seed); order = avail[:]; rng.shuffle(order)
        sel = []
        for k in order:
            sel.append(k)
            if sum(cnt[c] for c in sel) >= budget: break
        return sel
    if strategy == "big":
        order = sorted(avail, key=lambda k: -cnt[k]); sel = []
        for k in order:
            sel.append(k)
            if sum(cnt[c] for c in sel) >= budget: break
        return sel
    # coverage-based greedy strategies
    sel, best = [], np.full(len(ks), -np.inf)
    while True:
        cand = [k for k in avail if k not in sel]
        if not cand: break
        gains = []
        for k in cand:
            j = ks.index(k)
            g = float(np.mean(np.maximum(best, C[:, j]))) - (float(np.mean(best)) if sel else -1.0)
            if strategy == "cb":     score = g / max(1.0, cnt[k])          # gain per image
            elif strategy == "wcov": score = g * np.log1p(cnt[k])          # weighted by class size
            else:                    score = g                            # pure coverage
            gains.append((score, k))
        k = max(gains)[1]
        sel.append(k); best = np.maximum(best, C[:, ks.index(k)])
        if sum(cnt[c] for c in sel) >= budget: break
    return sel


def write_manifest(ds, alloc, tag, seed):
    """Manifiesto con submuestreo EXACTO por clase (test intacto, bonafide intacto)."""
    src = os.path.join(MAN_SRC, f"{ds}_subtype.csv")
    dst = os.path.join(MAN_OUT, f"{ds}__{tag}.csv")
    rows_by = {k: [] for k in alloc}
    head, keep = None, []
    with open(src) as f:
        rd = csv.DictReader(f); head = rd.fieldnames
        for r in rd:
            if r["split"] == "test" or r["subtype"] == "live":
                keep.append([r[c] for c in head])
            elif r["split"] == "train" and r["subtype"] in rows_by:
                rows_by[r["subtype"]].append([r[c] for c in head])
            elif r["split"] == "dev":
                keep.append([r[c] for c in head])
    rng = random.Random(seed)
    for k, n in alloc.items():
        pool = rows_by[k]; rng.shuffle(pool); keep += pool[:n]
    with open(dst, "w", newline="") as g:
        w = csv.writer(g); w.writerow(head); w.writerows(keep)
    return dst


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    ks, C = load_axes(ds)
    cnt = counts(ds)
    jobs, meta = [], {}
    print(f"== FIXED image budget [{ds}] ==")
    for B in BUDGETS:
        print(f"\n  --- budget of {B} attack images ---")
        for strat in ["cov", "cb", "wcov", "big", "rand"]:
            sel = pick(strat, ks, C, cnt, B, seed=0)
            alloc = waterfill(sel, cnt, B)
            if alloc is None:
                print(f"    {strat:5s}: cannot reach the budget, skipped"); continue
            cv = cov_of(ks, C, sel)
            tag = f"H{B}{strat}"
            # dedupe: if another strategy produced EXACTLY the same allocation, do not retrain
            sig_alloc = tuple(sorted(alloc.items()))
            dup = next((t for t, mm in meta.items() if mm["budget"] == B
                        and tuple(sorted(mm["alloc"].items())) == sig_alloc), None)
            if dup:
                print(f"    {strat:5s} == {meta[dup]['strategy']} (same allocation) -> reuses {dup}")
                meta[tag] = {"budget": B, "strategy": strat, "duplicate_of": dup,
                             "subset": sel, "alloc": alloc, "coverage": cov_of(ks, C, sel),
                             "n_imgs": sum(alloc.values())}
                continue
            meta[tag] = {"budget": B, "strategy": strat, "subset": sel,
                         "alloc": alloc, "coverage": cv, "n_imgs": sum(alloc.values())}
            print(f"    {strat:5s} cov={cv:.3f} m={len(sel)} n={sum(alloc.values())}  {', '.join(sel)}")
            for s in range(SEEDS):
                path = write_manifest(ds, alloc, f"{tag}s{s}", seed=100 + s)
                jobs.append({
                    "block": "H-budget", "label": f"std-{tag}-s{s}", "n_iters": len(sel) + 1,
                    "est_min": 45, "script": "ibt_generic.py",
                    "sig": f"--manifest {path} --dataset {ds} --design within --order standard --seed {s} ",
                    "args": (f"--manifest {path} --dataset {ds} --design within --order standard "
                             f"--seed {s} --model resnet50 --epochs_per_iter 3 --cap_per_class 100000 "
                             f"--batch_size 64 --tag std-{tag}"),
                })
    mp = os.path.join(ART, "budget_cells.json")
    allm = json.load(open(mp)) if os.path.isfile(mp) else {}
    allm[ds] = meta
    json.dump(allm, open(mp, "w"), indent=1)
    jp = os.path.join(ART, "jobs.json")
    cur = json.load(open(jp)); have = {j["label"] for j in cur}
    add = [j for j in jobs if j["label"] not in have]
    cur.extend(add); json.dump(cur, open(jp, "w"), indent=1)
    print(f"\n  +{len(add)} jobs (~{len(add)*45/60:.0f} GPU-h) -> jobs.json (total {len(cur)})")


if __name__ == "__main__":
    main()
