"""Exp 1e: scenario-contamination sensitivity. Recompute the Exp 1b revision-direction contrasts
keeping only judge-adjacent pairs whose spanning text contains no word "if" (conditional scenarios
like "if 250 spots ... if 350 spots" are the main labeling ambiguity). Prints per-model contrasts and
names any contrast that loses the motivated direction. Output: analysis/if_sensitivity.log
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import exp1b_revisions as e1b
from exp1_backtracking import RUNS, split_sentences, align_trajectory

e1b.N_BOOT = 800
IF = re.compile(r"\bif\b", re.I)

def pairs_with_if(reasoning, traj, threshold):
    sentences = split_sentences(reasoning)
    hits, _ = align_trajectory(sentences, traj, threshold)
    out = []
    for (s1, o1, v1, t1), (s2, o2, v2, t2) in zip(hits, hits[1:]):
        if t2 != t1 + 1:
            continue
        between = " ".join(sentences[s1:s2 + 1])
        out.append({"v1": v1, "v2": v2, "side": "above" if v1 > threshold else "below",
                    "down": v2 < v1, "same": v2 == v1, "noif": not IF.search(between)})
    return out

def load(rd, cond, trajs, thr):
    rows = json.load(open(os.path.join(rd, f"{cond}.json")))["rows"]
    out = []
    for row, traj in zip(rows, trajs[cond]):
        r = row.get("reasoning") or ""
        if not r or not traj:
            continue
        p = pairs_with_if(r, traj, thr)
        if p:
            out.append({"pairs": p})
    return out

if __name__ == "__main__":
    flips = []
    for d in sorted(os.listdir(RUNS)):
        rd = os.path.join(RUNS, d)
        if not os.path.exists(f"{rd}/threshold.json"):
            continue
        thr = json.load(open(f"{rd}/threshold.json"))["threshold"]
        trajs = json.load(open(f"{rd}/trajectories.json"))
        c = {k: load(rd, k, trajs, thr) for k in ["below_good", "above_good"]}
        tot = sum(len(r["pairs"]) for k in c for r in c[k])
        kept = sum(p["noif"] for k in c for r in c[k] for p in r["pairs"])
        d1 = e1b.boot_diff(c["below_good"], c["above_good"], "above", "down", subset="noif")
        d2u = e1b.boot_diff(c["above_good"], c["below_good"], "below", "down", subset="noif")
        d2 = None if d2u is None else (-d2u[0], -d2u[2], -d2u[1])
        for side, dd in (("above", d1), ("below", d2)):
            tag = "n/a" if dd is None else ("MOTIVATED" if dd[0] > 0 else "FLIPPED")
            if dd is not None and dd[0] <= 0:
                flips.append((d, side, dd))
            print(f"{d:40s} {side:5s} noif Δ {e1b.fmt(dd)}  {tag}  pairs kept {kept}/{tot}")
    print("\nFLIPPED contrasts:", [(m, s, e1b.fmt(dd)) for m, s, dd in flips])
