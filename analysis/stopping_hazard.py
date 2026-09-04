"""Motivated stopping (Kruglanski freezing / Ditto & Lopez quantity of processing, ported).

For each judge-extracted estimate trajectory, every non-final estimate is a "continue" event and the
final one a "stop" event. Hazard h(side, cond) = P(stop | current estimate on side). The valence swap
kills anchoring exactly as in R2: motivated stopping predicts h(above | above_good) > h(above |
below_good) and the mirror below. Cluster bootstrap over rollouts. Runs on the 10-model dataset and
the release transfer_runs. Writes analysis/stopping_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS

ROOT = os.path.join(os.path.dirname(__file__), "..")
TR = os.path.join(ROOT, "transfer_runs")
RNG = np.random.default_rng(0)
N_BOOT = 1000

def rollout_events(traj, thr):
    """[(side, stopped)] per estimate in one rollout."""
    if not traj or len(traj) < 2:
        return []
    out = []
    for i, v in enumerate(traj):
        if v == thr:
            continue
        out.append(("above" if v > thr else "below", i == len(traj) - 1))
    return out

def load(run_dir):
    thr = json.load(open(os.path.join(run_dir, "threshold.json")))["threshold"]
    trajs = json.load(open(os.path.join(run_dir, "trajectories.json")))
    conds = {}
    for c in ["baseline", "below_good", "above_good"]:
        if c not in trajs: continue
        rolls = [rollout_events(t, thr) for t in trajs[c] if t]
        conds[c] = [r for r in rolls if r]
    return conds

def hazard(rolls, side):
    k = n = 0
    for r in rolls:
        for s, stop in r:
            if s == side:
                n += 1; k += stop
    return k, n

def boot_contrast(ra, rb, side):
    """h(side|ra) - h(side|rb), cluster bootstrap over rollouts."""
    diffs = []
    for _ in range(N_BOOT):
        sa = [ra[i] for i in RNG.integers(0, len(ra), len(ra))]
        sb = [rb[i] for i in RNG.integers(0, len(rb), len(rb))]
        ka, na = hazard(sa, side); kb, nb = hazard(sb, side)
        if na >= 5 and nb >= 5:
            diffs.append(ka / na - kb / nb)
    if len(diffs) < 100: return None
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

def analyse(name, run_dir, rows):
    conds = load(run_dir)
    if "below_good" not in conds or "above_good" not in conds: return
    if min(len(conds["below_good"]), len(conds["above_good"])) < 15: return
    # motivated stopping: stop more readily on the GOOD side
    d1 = boot_contrast(conds["above_good"], conds["below_good"], "above")   # above good vs above bad
    d2 = boot_contrast(conds["below_good"], conds["above_good"], "below")   # below good vs below bad
    ha = hazard(conds["above_good"], "above"); hb = hazard(conds["below_good"], "above")
    for side, d in (("above", d1), ("below", d2)):
        if d is None: continue
        rows.append({"model": name, "side": side, "d": d[0], "lo": d[1], "hi": d[2]})
    f = lambda d: "n/a" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]"
    print(f"{name:44s} Δh(above) {f(d1)}  Δh(below) {f(d2)}  raw h(above): good {ha[0]}/{ha[1]} bad {hb[0]}/{hb[1]}")

if __name__ == "__main__":
    rows = []
    print("== 10-model dataset ==")
    for d in sorted(os.listdir(RUNS)):
        rd = os.path.join(RUNS, d)
        if os.path.exists(os.path.join(rd, "threshold.json")):
            analyse(d.split("_")[0], rd, rows)
    print("== release (transfer_runs) ==")
    rel = []
    for d in sorted(os.listdir(TR)):
        rd = os.path.join(TR, d)
        if os.path.isdir(rd) and os.path.exists(os.path.join(rd, "threshold.json")):
            analyse(d, rd, rel)
    pos = sum(r["d"] > 0 for r in rows); sig = sum(r["lo"] > 0 for r in rows)
    print(f"\nDATASET SUMMARY: {pos}/{len(rows)} motivated-stopping direction, {sig} sig, {sum(r['hi']<0 for r in rows)} sig reversed")
    posr = sum(r["d"] > 0 for r in rel); sigr = sum(r["lo"] > 0 for r in rel)
    print(f"RELEASE SUMMARY: {posr}/{len(rel)} motivated-stopping direction, {sigr} sig, {sum(r['hi']<0 for r in rel)} sig reversed")
    json.dump({"dataset": rows, "release": rel}, open(os.path.join(ROOT, "analysis", "stopping_results.json"), "w"), indent=1)
    print("STOPPING_DONE")

def strata(out_path=None):
    """Absolute-estimate-index-stratified contrasts over dataset + release (seeded)."""
    rng = np.random.default_rng(1)
    def ev(traj, thr):
        if not traj or len(traj) < 2: return []
        return [("above" if v > thr else "below", i == len(traj) - 1, i) for i, v in enumerate(traj) if v != thr]
    def hz(rolls, side, ilo, ihi):
        k = n = 0
        for r in rolls:
            for sd, stop, i in r:
                if sd == side and ilo <= i < ihi: n += 1; k += stop
        return k, n
    def con(ra, rb, side, ilo, ihi, B=400):
        ds = []
        for _ in range(B):
            sa = [ra[i] for i in rng.integers(0, len(ra), len(ra))]
            sb = [rb[i] for i in rng.integers(0, len(rb), len(rb))]
            ka, na = hz(sa, side, ilo, ihi); kb, nb = hz(sb, side, ilo, ihi)
            if na >= 5 and nb >= 5: ds.append(ka / na - kb / nb)
        if len(ds) < 100: return None
        return np.mean(ds), np.percentile(ds, 2.5), np.percentile(ds, 97.5)
    def loadc(rd):
        thr = json.load(open(f"{rd}/threshold.json"))["threshold"]
        trajs = json.load(open(f"{rd}/trajectories.json"))
        return {c: [e for e in (ev(t, thr) for t in trajs.get(c, []) if t) if e] for c in ["below_good", "above_good"]}
    strata_edges = [(1, 4), (4, 8), (8, 200)]
    res = {i: [0, 0, 0, 0] for i in range(3)}
    dirs = [os.path.join(RUNS, d) for d in sorted(os.listdir(RUNS)) if os.path.exists(os.path.join(RUNS, d, "threshold.json"))]
    dirs += [os.path.join(TR, d) for d in sorted(os.listdir(TR)) if os.path.isdir(os.path.join(TR, d)) and os.path.exists(os.path.join(TR, d, "threshold.json"))]
    for rd in dirs:
        c = loadc(rd)
        if min(len(c["below_good"]), len(c["above_good"])) < 15: continue
        for si, (ilo, ihi) in enumerate(strata_edges):
            for side, (ra, rb) in (("above", (c["above_good"], c["below_good"])), ("below", (c["below_good"], c["above_good"]))):
                d = con(ra, rb, side, ilo, ihi)
                if d is None: continue
                res[si][3] += 1; res[si][0] += int(d[0] > 0); res[si][1] += int(d[1] > 0); res[si][2] += int(d[2] < 0)
    out = {}
    for si, (ilo, ihi) in enumerate(strata_edges):
        p, sg, ng, n = res[si]
        out[f"[{ilo},{ihi})"] = {"motivated": p, "n": n, "sig": sg, "sig_reversed": ng}
        print(f"index [{ilo},{ihi}): {p}/{n} motivated, {sg} sig, {ng} sig reversed")
    json.dump(out, open(out_path or os.path.join(ROOT, "analysis", "stopping_strata.json"), "w"), indent=1)

