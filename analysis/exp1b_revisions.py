"""Experiment 1b: Revision DIRECTION and threshold attention.

For consecutive in-CoT estimate pairs (v_i -> v_{i+1}):
  - P(revision moves DOWN | current side)  per condition, vs baseline mean-reversion.
    Motivated prediction at side=above: below_good (above=bad) pulls DOWN more than
    above_good (above=good), with baseline in between.
  - Same at side=below for P(move UP).
  - Magnitude: median signed log-ratio by cell.
  - Backtrack-linked revisions: pairs where a STRONG marker appears between the
    two estimates (inclusive of the arrival sentence).
  - Threshold attention: P(sentence mentions threshold/bet | side) per condition.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import (RUNS, split_sentences, numbers_in, close, STRONG,
                               align_trajectory)

RNG = np.random.default_rng(1)
N_BOOT = 2000

def thr_regexes(threshold):
    pats = [r"\bthreshold\b", r"\bbet\b", r"\bwager\b", r"donat", r"good cause", r"bad cause"]
    t = int(threshold)
    comma = f"{t:,}"
    pats.append(re.escape(comma))
    if t % 1_000_000 == 0:
        m = t // 1_000_000
        pats.append(rf"\b{m}\s*(million|m\b)")
    elif t % 100_000 == 0:
        m = t / 1_000_000
        pats.append(rf"\b{m:g}\s*(million|m\b)")
    return re.compile("|".join(pats), re.IGNORECASE)

def rollout_pairs(reasoning, traj, threshold, thr_re):
    """Returns (pairs, sent_flags). pairs: list of dicts per consecutive estimate
    pair. sent_flags: per-sentence (side, thr_mention, backtrack) after first est."""
    sentences = split_sentences(reasoning)
    hits, _ = align_trajectory(sentences, traj, threshold)
    if len(hits) < 2:
        return [], []
    pairs = []
    for (s1, o1, v1, t1), (s2, o2, v2, t2) in zip(hits, hits[1:]):
        if t2 != t1 + 1:  # only judge-adjacent estimates form a revision step
            continue
        between = " ".join(sentences[s1:s2 + 1])
        pairs.append({
            "v1": v1, "v2": v2,
            "side": "above" if v1 > threshold else "below",
            "down": v2 < v1,
            "same": v2 == v1,
            "logr": float(np.log(v2 / v1)) if v1 > 0 and v2 > 0 else 0.0,
            "backtracked": bool(STRONG.search(between)),
            "thr_mention": bool(thr_re.search(between)),
            "crossed": (v1 > threshold) != (v2 > threshold),
        })
    # per-sentence threshold-attention flags
    est_by_sent = defaultdict(list)
    for si, off, val, _tj in hits:
        est_by_sent[si].append((off, val))
    flags, side = [], None
    first = hits[0][0]
    for si, sent in enumerate(sentences):
        if est_by_sent.get(si):
            side = "above" if est_by_sent[si][-1][1] > threshold else "below"
        if si > first and side is not None:
            flags.append((side, bool(thr_re.search(sent)), bool(STRONG.search(sent))))
    return pairs, flags

def load_condition(run_dir, cond, trajs, threshold, thr_re):
    rows = json.load(open(os.path.join(run_dir, f"{cond}.json")))["rows"]
    out = []
    for row, traj in zip(rows, trajs[cond]):
        r = row.get("reasoning") or ""
        if not r or not traj:
            continue
        pairs, flags = rollout_pairs(r, traj, threshold, thr_re)
        if pairs:
            out.append({"pairs": pairs, "flags": flags})
    return out

def cell_rate(rollouts, side, key="down", subset=None):
    k = n = 0
    for ro in rollouts:
        for p in ro["pairs"]:
            if p["side"] != side or p["same"]:
                continue
            if subset and not p[subset]:
                continue
            n += 1
            k += bool(p[key])
    return k, n

def boot_diff(ra, rb, side, key="down", subset=None):
    diffs = []
    for _ in range(N_BOOT):
        sa = [ra[i] for i in RNG.integers(0, len(ra), len(ra))]
        sb = [rb[i] for i in RNG.integers(0, len(rb), len(rb))]
        ka, na = cell_rate(sa, side, key, subset)
        kb, nb = cell_rate(sb, side, key, subset)
        if na >= 5 and nb >= 5:
            diffs.append(ka / na - kb / nb)
    if len(diffs) < 100:
        return None
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))

def fmt(x):
    return "n/a" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]"

def analyse(run_dir):
    thr = json.load(open(os.path.join(run_dir, "threshold.json")))["threshold"]
    trajs = json.load(open(os.path.join(run_dir, "trajectories.json")))
    thr_re = thr_regexes(thr)
    conds = {c: load_condition(run_dir, c, trajs, thr, thr_re)
             for c in ["baseline", "below_good", "above_good"]}
    res = {"threshold": thr}
    print(f"  {'cond':11s} {'P(down|above)':>16s} {'P(up|below)':>16s} {'medlogr(ab)':>12s} {'medlogr(be)':>12s} pairs")
    for cond, ros in conds.items():
        ka, na = cell_rate(ros, "above", "down")
        kb, nb = cell_rate(ros, "below", "down")
        lr_a = [p["logr"] for ro in ros for p in ro["pairs"] if p["side"] == "above" and not p["same"]]
        lr_b = [p["logr"] for ro in ros for p in ro["pairs"] if p["side"] == "below" and not p["same"]]
        pa = ka / na if na else float("nan")
        pb = 1 - kb / nb if nb else float("nan")
        print(f"  {cond:11s} {pa:>10.3f} n={na:<5d} {pb:>10.3f} n={nb:<5d} "
              f"{np.median(lr_a) if lr_a else float('nan'):>+12.4f} {np.median(lr_b) if lr_b else float('nan'):>+12.4f} "
              f"{sum(len(ro['pairs']) for ro in ros)}")
        res[cond] = {"p_down_above": pa, "n_above": na, "p_up_below": pb, "n_below": nb,
                     "med_logr_above": float(np.median(lr_a)) if lr_a else None,
                     "med_logr_below": float(np.median(lr_b)) if lr_b else None}
    # Key contrasts (motivated direction predictions):
    # at side=above: P(down) below_good > above_good        -> d1 > 0
    # at side=below: P(up)   above_good > below_good        -> d2 > 0  (P(up)=1-P(down))
    d1 = boot_diff(conds["below_good"], conds["above_good"], "above", "down")
    d2u = boot_diff(conds["above_good"], conds["below_good"], "below", "down")
    d2 = None if d2u is None else (-d2u[0], -d2u[2], -d2u[1])
    d1b = boot_diff(conds["below_good"], conds["above_good"], "above", "down", subset="backtracked")
    d2ub = boot_diff(conds["above_good"], conds["below_good"], "below", "down", subset="backtracked")
    d2b = None if d2ub is None else (-d2ub[0], -d2ub[2], -d2ub[1])
    print(f"  MOTIVATED-DIRECTION  Δ P(down|above) bg−ag: {fmt(d1)}   Δ P(up|below) ag−bg: {fmt(d2)}")
    print(f"  ...backtracked-only  Δ P(down|above) bg−ag: {fmt(d1b)}  Δ P(up|below) ag−bg: {fmt(d2b)}")
    res["d_down_above_bg_minus_ag"] = d1; res["d_up_below_ag_minus_bg"] = d2
    res["d_down_above_bt_only"] = d1b; res["d_up_below_bt_only"] = d2b
    # Threshold attention per side
    print(f"  {'cond':11s} P(thr-mention|above)  P(thr-mention|below)  P(backtrack|thr-mention)")
    for cond, ros in conds.items():
        fa = [(t, b) for ro in ros for (s, t, b) in ro["flags"] if s == "above"]
        fb = [(t, b) for ro in ros for (s, t, b) in ro["flags"] if s == "below"]
        tm = [(b) for ro in ros for (s, t, b) in ro["flags"] if t]
        pa = np.mean([t for t, _ in fa]) if fa else float("nan")
        pb = np.mean([t for t, _ in fb]) if fb else float("nan")
        pbt = np.mean(tm) if tm else float("nan")
        res[cond].update({"p_thr_above": float(pa), "p_thr_below": float(pb),
                          "p_bt_given_thr": float(pbt)})
        print(f"  {cond:11s} {pa:>12.3f} {pb:>21.3f} {pbt:>22.3f}")
    return res

def main():
    models = sorted(os.listdir(RUNS))
    if len(sys.argv) > 1:
        models = [m for m in models if any(a in m for a in sys.argv[1:])]
    results = {}
    for m in models:
        rd = os.path.join(RUNS, m)
        if not os.path.isdir(rd):
            continue
        print(f"\n=== {m} ===")
        try:
            results[m] = analyse(rd)
        except Exception as e:
            print("  FAILED", repr(e))
    with open(os.path.join(os.path.dirname(__file__), "exp1b_results.json"), "w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    main()
