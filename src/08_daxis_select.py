#!/usr/bin/env python3
"""
DAXIS-SELECT - choosing PAIs by maximising ANGULAR COVERAGE (facility location).

Motivation (07_coverage_law): what predicts performance is not the alignment or the dispersion of
the subset but the coverage  cov(S) = mean_k max_{j in S} cos(d_k, d_j).

That objective is monotone and SUBMODULAR, so greedy carries a (1-1/e) guarantee. 'Aligned' and
'diverse' are heuristics that approximate coverage at opposite ends of the budget; maximising it
directly should dominate at every m - a falsifiable prediction.

The prediction turned out false, for a reason worth recording. See 13_budget_select: the objective
is blind to how much data each PAI carries, so at equal PAI count it can pick angularly ideal but
data-poor sets.

Usage: python 08_daxis_select.py [HQ-WMCA]
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from config import ART, RES_DAXIS, RES_CURRICULUM, FIG_DIR, OUT, FAS_DATA_ROOT, MANIFEST_DIR  # noqa
MAN_OUT = os.path.join(HERE, "manifests")
MAN_SRC = os.path.join(HERE, "..", "fas_benchmark", "manifests")
os.makedirs(MAN_OUT, exist_ok=True)


def greedy_coverage(ks, C, m):
    """Facility location: en cada paso añade el eje que más sube mean_k max_{j∈S} cos(d_k,d_j)."""
    n = len(ks)
    sel, best = [], np.full(n, -np.inf)
    for _ in range(m):
        gains = []
        for j in range(n):
            if j in sel: gains.append(-np.inf); continue
            gains.append(float(np.mean(np.maximum(best, C[:, j]))))
        j = int(np.argmax(gains))
        sel.append(j); best = np.maximum(best, C[:, j])
    return [ks[j] for j in sel], float(np.mean(best))


def cov_of(ks, C, subset):
    idx = [ks.index(k) for k in subset]
    return float(np.mean([max(C[k, j] for j in idx) for k in range(len(ks))]))


def write_manifest(ds, keep, tag):
    src = os.path.join(MAN_SRC, f"{ds}_subtype.csv")
    dst = os.path.join(MAN_OUT, f"{ds}__{tag}.csv")
    keep = set(keep) | {"live"}
    n = 0
    with open(src) as f, open(dst, "w", newline="") as g:
        rd = csv.DictReader(f); w = csv.writer(g); w.writerow(rd.fieldnames)
        for r in rd:
            # TEST is left intact (we always evaluate on all classes); only train/dev are filtered
            if r["split"] == "test" or r["subtype"] in keep:
                w.writerow([r[c] for c in rd.fieldnames]); n += 1
    return dst, n


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "HQ-WMCA"
    z = np.load(os.path.join(ART, f"axes_{ds}.npz".replace("+", "plus")), allow_pickle=True)
    ks = [str(k) for k in z["ks"]]; C = z["C"]
    blob = json.load(open(os.path.join(ART, "orders.json")))[ds]
    picks = blob["picks"]

    print(f"== DAXIS-select (facility location) [{ds}] ==")
    print(f"  {'m':>2s} {'coverage':>9s}  subset   |  cov(aligned) cov(diverse)")
    newpicks = {}
    jobs = []
    for m in (2, 4, 6, 8):
        S, cov = greedy_coverage(ks, C, m)
        covA = cov_of(ks, C, picks["L3_topm_aligned"][str(m)])
        covD = cov_of(ks, C, picks["L3b_diverse"][str(m)])
        newpicks[str(m)] = S
        star = " <-- wins" if cov > max(covA, covD) else ""
        print(f"  {m:2d} {cov:9.3f}  {', '.join(S)}")
        print(f"     {'':9s}  vs aligned {covA:.3f} · diverse {covD:.3f}{star}")
        tag = f"F1sel{m}"
        path, n = write_manifest(ds, S, tag)
        for s in range(5):
            jobs.append({
                "block": "F-select", "label": f"std-{tag}-s{s}", "n_iters": 1, "est_min": 40,
                "script": "ibt_generic.py",
                "sig": f"--manifest {path} --dataset {ds} --design within --order standard --seed {s} ",
                "args": (f"--manifest {path} --dataset {ds} --design within --order standard "
                         f"--seed {s} --model resnet50 --epochs_per_iter 3 --cap_per_class 4000 "
                         f"--batch_size 64 --tag std-{tag}"),
            })
    # n_iters real = 1 etapa (standard sobre m clases -> el entrenador hace m+1 iters);
    # lo dejamos al valor que produce ibt_generic: len(order)+1
    for m, j in zip([2]*5 + [4]*5 + [6]*5 + [8]*5, jobs):
        j["n_iters"] = m + 1

    blob["picks"]["F_daxis_select"] = newpicks
    allo = json.load(open(os.path.join(ART, "orders.json"))); allo[ds] = blob
    json.dump(allo, open(os.path.join(ART, "orders.json"), "w"), indent=1, ensure_ascii=False)

    jp = os.path.join(ART, "jobs.json")
    cur = json.load(open(jp))
    have = {j["label"] for j in cur}
    add = [j for j in jobs if j["label"] not in have]
    cur.extend(add)
    json.dump(cur, open(jp, "w"), indent=1)
    print(f"\n  +{len(add)} jobs appended to jobs.json (total {len(cur)})")
    print("  (the scheduler picks them up on its next cycle)")


if __name__ == "__main__":
    main()
