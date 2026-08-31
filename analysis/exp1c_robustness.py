"""Experiment 1c: Robustness of the motivated revision-direction effect.

Confound 1 (distance): conditional on side, |log(v_i/threshold)| may differ by
condition; larger excursions mean-revert harder. Fix: stratify pairs by pooled
quartiles of log-distance within side, compute the contrast within strata,
average strata (equal weights), cluster-bootstrap over rollouts.

Confound 2 (position): revision behavior drifts over the trace, and conditions
occupy sides at different times. Fix: same, stratified by pair-position third.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from exp1b_revisions import rollout_pairs, thr_regexes
from exp1_backtracking import RUNS

RNG = np.random.default_rng(2)
N_BOOT = 1000

def load(run_dir, cond, trajs, thr, thr_re):
    rows = json.load(open(os.path.join(run_dir, f"{cond}.json")))["rows"]
    out = []
    for row, traj in zip(rows, trajs[cond]):
        r = row.get("reasoning") or ""
        if not r or not traj:
            continue
        pairs, _ = rollout_pairs(r, traj, thr, thr_re)
        if not pairs:
            continue
        n = len(pairs)
        for j, p in enumerate(pairs):
            p["pos"] = j / max(1, n - 1)
            p["dist"] = abs(np.log(max(p["v1"], 1) / thr))
        out.append(pairs)
    return out

def stratified_diff(ra, rb, side, key, strata_fn, edges):
    """Equal-weighted mean over strata of rate_a - rate_b."""
    def rates(rollouts):
        k = np.zeros(len(edges) + 1); n = np.zeros(len(edges) + 1)
        for ro in rollouts:
            for p in ro:
                if p["side"] != side or p["same"]:
                    continue
                s = int(np.searchsorted(edges, strata_fn(p)))
                n[s] += 1; k[s] += bool(p[key])
        return k, n
    ka, na = rates(ra); kb, nb = rates(rb)
    ok = (na >= 5) & (nb >= 5)
    if not ok.any():
        return None
    return float(np.mean((ka[ok] / na[ok]) - (kb[ok] / nb[ok])))

def boot(ra, rb, side, key, strata_fn, edges):
    pt = stratified_diff(ra, rb, side, key, strata_fn, edges)
    if pt is None:
        return None
    ds = []
    for _ in range(N_BOOT):
        sa = [ra[i] for i in RNG.integers(0, len(ra), len(ra))]
        sb = [rb[i] for i in RNG.integers(0, len(rb), len(rb))]
        d = stratified_diff(sa, sb, side, key, strata_fn, edges)
        if d is not None:
            ds.append(d)
    return pt, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def fmt(x):
    return "  n/a " if x is None else f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]"

def main():
    models = sorted(os.listdir(RUNS))
    if len(sys.argv) > 1:
        models = [m for m in models if any(a in m for a in sys.argv[1:])]
    results = {}
    for m in models:
        rd = os.path.join(RUNS, m)
        if not os.path.isdir(rd):
            continue
        thr = json.load(open(os.path.join(rd, "threshold.json")))["threshold"]
        trajs = json.load(open(os.path.join(rd, "trajectories.json")))
        thr_re = thr_regexes(thr)
        conds = {c: load(rd, c, trajs, thr, thr_re) for c in ["below_good", "above_good"]}
        # attach v1-distance: rollout_pairs stores logr but not v1; recompute dist from
        # sequence: reconstruct via logr chain is fragile -> re-derive in rollout_pairs?
        # rollout_pairs has no v1; patch: use 'crossed' + logr not enough. Use dist proxy:
        # we re-run alignment here minimally by storing dist at build time instead.
        print(f"\n=== {m} (thr {thr:,}) ===")
        res = {}
        for (side, key, name, a, b) in [
            ("above", "down", "P(down|above) bg-ag", "below_good", "above_good"),
            ("below", "up",   "P(up|below)  ag-bg", "above_good", "below_good"),
        ]:
            for ro in conds[a] + conds[b]:
                for p in ro:
                    p["up"] = (not p["down"]) and not p["same"]
            # distance strata: pooled quartiles of dist within side
            dists = [p["dist"] for c in (a, b) for ro in conds[c] for p in ro
                     if p["side"] == side and not p["same"] and p.get("dist") is not None]
            if not dists:
                print(f"  {name}: no distance data"); continue
            dq = np.percentile(dists, [25, 50, 75])
            r_dist = boot(conds[a], conds[b], side, key, lambda p: p["dist"], dq)
            r_pos = boot(conds[a], conds[b], side, key, lambda p: p["pos"], np.array([1/3, 2/3]))
            print(f"  {name}:  distance-stratified {fmt(r_dist)}   position-stratified {fmt(r_pos)}")
            res[name] = {"dist": r_dist, "pos": r_pos}
        results[m] = res
    with open(os.path.join(os.path.dirname(__file__), "exp1c_results.json"), "w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    main()
