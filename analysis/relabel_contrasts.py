"""R2 contrasts restricted to judged-REVISION pairs (from relabel_pairs.py output). Seeded."""
import json, sys, collections, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from relabel_pairs import collect_pairs

ROOT = os.path.join(os.path.dirname(__file__), "..")
rng = np.random.default_rng(2)
labels = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(os.path.join(ROOT, "exp2_out", "pair_labels.jsonl"))}
rolls = collections.defaultdict(list)
for p in collect_pairs():
    run, cond, ri, t1 = p["id"].split("|")
    rolls[(run, cond, ri)].append({"side": "above" if p["v1"] > p["thr"] else "below",
                                   "down": p["v2"] < p["v1"], "lab": labels.get(p["id"])})
bym = collections.defaultdict(lambda: collections.defaultdict(list))
for (run, cond, ri), v in rolls.items(): bym[run][cond].append(v)

def rate(rl, side, only_rev):
    k = n = 0
    for r in rl:
        for p in r:
            if p["side"] != side: continue
            if only_rev and p["lab"] != "REVISION": continue
            n += 1; k += p["down"]
    return k, n

def boot(ra, rb, side, only_rev, B=800):
    ds = []
    for _ in range(B):
        sa = [ra[i] for i in rng.integers(0, len(ra), len(ra))]
        sb = [rb[i] for i in rng.integers(0, len(rb), len(rb))]
        ka, na = rate(sa, side, only_rev); kb, nb = rate(sb, side, only_rev)
        if na >= 5 and nb >= 5: ds.append(ka / na - kb / nb)
    if len(ds) < 100: return None
    return np.mean(ds), np.percentile(ds, 2.5), np.percentile(ds, 97.5)

def contrast(conds, side, only_rev):
    if side == "above":
        return boot(conds["below_good"], conds["above_good"], "above", only_rev)
    d = boot(conds["above_good"], conds["below_good"], "below", only_rev)
    return None if d is None else (-d[0], -d[2], -d[1])

if __name__ == "__main__":
    out = []
    tot = pos = sig = 0
    for run, conds in sorted(bym.items()):
        for side in ("above", "below"):
            dr = contrast(conds, side, True)
            if dr is None: continue
            tot += 1; pos += dr[0] > 0; sig += dr[1] > 0
            out.append({"model": run.split("_")[0], "side": side, "d": float(dr[0]), "lo": float(dr[1]), "hi": float(dr[2])})
            print(f"{run.split('_')[0]:22s} {side:5s} REVISION-only {dr[0]:+.3f} [{dr[1]:+.3f},{dr[2]:+.3f}]")
    print(f"\nREVISION-only: {pos}/{tot} motivated, {sig} sig")
    json.dump(out, open(os.path.join(ROOT, "analysis", "relabel_contrasts.json"), "w"), indent=1)
