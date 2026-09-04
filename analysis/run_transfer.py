"""Run the Exp 1b revision-direction contrasts over transfer_runs/ (authors' release,
9 questions x models) and summarize: fraction of contrasts in the motivated direction,
significant count, per-question breakdown."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import exp1_backtracking as e1
import exp1b_revisions as e1b

ROOT = os.path.join(os.path.dirname(__file__), "..")
TR = os.path.join(ROOT, "transfer_runs")
e1.RUNS = TR; e1b.RUNS = TR
e1b.N_BOOT = 800

rows = []
for d in sorted(os.listdir(TR)):
    rd = os.path.join(TR, d)
    if not os.path.isdir(rd): continue
    try:
        thr = json.load(open(f"{rd}/threshold.json"))["threshold"]
        trajs = json.load(open(f"{rd}/trajectories.json"))
        thr_re = e1b.thr_regexes(thr)
        conds = {c: e1b.load_condition(rd, c, trajs, thr, thr_re) for c in ["baseline", "below_good", "above_good"]}
        if min(len(conds["below_good"]), len(conds["above_good"])) < 15:
            print(f"{d}: too few usable rollouts ({len(conds['below_good'])}/{len(conds['above_good'])})"); continue
        d1 = e1b.boot_diff(conds["below_good"], conds["above_good"], "above", "down")
        d2u = e1b.boot_diff(conds["above_good"], conds["below_good"], "below", "down")
        d2 = None if d2u is None else (-d2u[0], -d2u[2], -d2u[1])
        model, q = d.split("__")
        for side, dd in (("above", d1), ("below", d2)):
            if dd is None: continue
            rows.append({"model": model, "question": q, "side": side, "d": dd[0], "lo": dd[1], "hi": dd[2]})
        print(f"{d:48s} Δabove {e1b.fmt(d1)}  Δbelow {e1b.fmt(d2)}")
    except Exception as ex:
        print(f"{d}: FAILED {ex!r}")

json.dump(rows, open(os.path.join(ROOT, "analysis", "transfer_results.json"), "w"), indent=1)
n = len(rows); pos = sum(r["d"] > 0 for r in rows); sig = sum(r["lo"] > 0 for r in rows); neg_sig = sum(r["hi"] < 0 for r in rows)
print(f"\nTRANSFER SUMMARY: {pos}/{n} contrasts motivated direction, {sig} significant, {neg_sig} significant reversals")
byq = {}
for r in rows:
    byq.setdefault(r["question"], []).append(r["d"] > 0)
for q, v in sorted(byq.items()):
    print(f"  {q:11s} {sum(v)}/{len(v)} motivated")
