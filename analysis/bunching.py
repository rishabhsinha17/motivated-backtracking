"""Bunching at the threshold (Burgstahler & Dichev 1997 / Kleven notch logic, valence-swap version).

The threshold is each model's baseline median, so the baseline mechanically piles mass at T and
cannot serve as the counterfactual. The clean counterfactual is the valence swap itself: the same T,
question and round-number structure appears in both incentive conditions and only the favored side
flips. With window w = 2% of T and asym(cond) = P(v in (T-w, T]) - P(v in (T, T+w]):
  b = asym(below_good) - asym(above_good)   (>0 = mass moves to whichever side pays)
Values come straight from the release cache (final answer parse, falling back to the last judge
trajectory estimate). Writes analysis/bunching_results.json.
"""
import json, os, collections
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
W = 0.02

import glob, hashlib, sys
sys.path.insert(0, os.path.dirname(__file__))
from exp2_resample import parse_final
from fig_transfer_testA import NUM as _N  # noqa: F401  (import check only)
CACHE = os.path.join(ROOT, "tfai_data", "final_data", "cache")
TRAJ = os.path.join(ROOT, "tfai_data", "final_data", "trajectories")
QUESTIONS = ["giraffes", "zills", "bridge", "crochet", "tbc", "maiden", "turns", "orangecars", "windowdays"]
tidx = {}
for f in glob.glob(os.path.join(TRAJ, "*.jsonl")):
    for line in open(f):
        j = json.loads(line); a = j.get("answer")
        if a and a != "NONE":
            try: tidx[j["r_hash"]] = [float(x) for x in str(a).split(",") if x.strip()]
            except ValueError: pass
data = collections.defaultdict(lambda: collections.defaultdict(list))
for model in sorted(os.listdir(CACHE)):
    for q in QUESTIONS:
        mdir = os.path.join(CACHE, model, f"v1_{q}_accurate")
        if not os.path.isdir(mdir): continue
        thr_counts = collections.Counter()
        rowbuf = collections.defaultdict(list)
        for cond in ["below_good", "above_good"]:
            for f in glob.glob(os.path.join(mdir, f"{cond}_*.jsonl")):
                for line in open(f).read().splitlines()[1:]:
                    j = json.loads(line)
                    if j.get("threshold") is not None: thr_counts[int(j["threshold"])] += 1
                    rowbuf[cond].append(j)
        if not thr_counts: continue
        t = thr_counts.most_common(1)[0][0]
        for cond, rl in rowbuf.items():
            for j in rl:
                if j.get("threshold") is not None and int(j["threshold"]) != t: continue
                v = parse_final(j.get("answer") or "")
                if v is None:
                    r = j.get("reasoning") or ""
                    tr = tidx.get(hashlib.sha256(r.encode()).hexdigest()[:12]) if r else None
                    v = tr[-1] if tr else None
                if v is not None: data[model][(q, cond)].append((v, t))

def asym(vals):
    under = np.mean([(t * (1 - W) < v <= t) for v, t in vals])
    over = np.mean([(t < v <= t * (1 + W)) for v, t in vals])
    return under - over

def bstat(cells):
    """cells: dict (q,cond)->list. Pool per condition across questions."""
    by = collections.defaultdict(list)
    for (q, cond), vals in cells.items(): by[cond] += vals
    if min(len(by.get(c, [])) for c in ["below_good", "above_good"]) < 50: return None
    def one(bg, ag): return asym(bg) - asym(ag)
    b = one(by["below_good"], by["above_good"])
    boots = []
    for _ in range(1000):
        s = {c: [by[c][i] for i in RNG.integers(0, len(by[c]), len(by[c]))] for c in by}
        boots.append(one(s["below_good"], s["above_good"]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    exact = np.mean([v == t for c in ["below_good", "above_good"] for v, t in by[c]])
    return b, float(lo), float(hi), float(exact), sum(len(by[c]) for c in ["below_good", "above_good"])

out = {}
print(f"{'model':44s} {'b (excess mass)':>22s}  exact-T  n")
for m in sorted(data):
    r = bstat(data[m])
    if r is None: continue
    b, lo, hi, exact, n = r
    flag = "BUNCHER" if lo > 0 else ("anti" if hi < 0 else "")
    out[m] = {"b": b, "lo": lo, "hi": hi, "exact_T": exact, "n": n}
    print(f"{m:44s} {b:+.3f} [{lo:+.3f},{hi:+.3f}]  {exact:.2f}  {n:5d}  {flag}")
sig = sum(1 for v in out.values() if v["lo"] > 0)
print(f"\n{sig}/{len(out)} models with significant excess mass at the threshold")
bs = [v["b"] for v in out.values()]; ex = [v["exact_T"] for v in out.values()]
rho, p = spearmanr(bs, ex)
print(f"bunching vs exact-T print rate: rho={rho:.2f} p={p:.4f}")
tr = json.load(open(os.path.join(ROOT, "analysis", "transfer_results.json")))
steer = collections.defaultdict(list)
for t in tr: steer[t["model"]].append(t["d"])
steer = {m: float(np.mean(v)) for m, v in steer.items()}
common = [m for m in out if m in steer]
rho2, p2 = spearmanr([out[m]["b"] for m in common], [steer[m] for m in common])
print(f"bunching vs steering strength (n={len(common)}): rho={rho2:.2f} p={p2:.4f}")
json.dump(out, open(os.path.join(ROOT, "analysis", "bunching_results.json"), "w"), indent=1)
print("BUNCHING_DONE")
