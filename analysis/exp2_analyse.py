"""Experiment 2 analysis: suppressed vs plain (matched forks) per arm/condition.

Primary: P(final estimate > threshold) suppressed vs plain, paired by trace.
Also: rejection-count distribution (resilience), parse-failure rates.
"""
import glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "exp2_out")
RUNS = os.path.join(HERE, "..", "value-leakage", "runs")
RNG = np.random.default_rng(3)

def load(path):
    per_trace = defaultdict(list)
    rej = []
    n_fail = n = 0
    seen = set()
    for line in open(path):
        j = json.loads(line)
        key = (j["trace"], j["sample"])
        if key in seen:   # duplicate rows from overlapping runners
            continue
        seen.add(key)
        n += 1
        if j.get("rejected") is not None:
            rej.append(j["rejected"])
        if j["final"] is None:
            n_fail += 1
            continue
        per_trace[j["trace"]].append(bool(j["above"]))
    return per_trace, rej, n_fail, n

def p_above(per_trace):
    vals = [v for t in per_trace.values() for v in t]
    return np.mean(vals) if vals else float("nan"), len(vals)

def paired_boot(a, b, n_boot=2000):
    traces = sorted(set(a) & set(b))
    if not traces:
        return None
    diffs = []
    for _ in range(n_boot):
        sample = RNG.choice(traces, size=len(traces), replace=True)
        da = [v for t in sample for v in a[t]]
        db = [v for t in sample for v in b[t]]
        diffs.append(np.mean(da) - np.mean(db))
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), len(traces))

for model in sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{OUT}/*arm*.jsonl")}):
    thr_dirs = [d for d in os.listdir(RUNS) if d.startswith(model)]
    thr = json.load(open(os.path.join(RUNS, thr_dirs[0], "threshold.json")))["threshold"]
    print(f"\n=== {model} (thr {thr:,}) ===")
    for cond in ["below_good", "above_good"]:
        for arm in "ABD":
            fs = os.path.join(OUT, f"{model}_{cond}_arm{arm}.jsonl")
            fp = os.path.join(OUT, f"{model}_{cond}_arm{arm}plain.jsonl")
            if not (os.path.exists(fs) and os.path.exists(fp)):
                continue
            s_pt, s_rej, s_fail, s_n = load(fs)
            p_pt, _, p_fail, p_n = load(fp)
            ps, ns = p_above(s_pt)
            pp, np_ = p_above(p_pt)
            pb = paired_boot(s_pt, p_pt)
            rejd = np.bincount(s_rej, minlength=7)[:7] if s_rej else []
            print(f" {cond} arm {arm}: P(above) suppressed={ps:.3f} (n={ns}) plain={pp:.3f} (n={np_})"
                  f"  paired Δ={pb[0]:+.3f} [{pb[1]:+.3f},{pb[2]:+.3f}] traces={pb[3]}"
                  f"  parsefail {s_fail}/{s_n} vs {p_fail}/{p_n}")
            if len(rejd):
                print(f"    rejection counts 0..6+: {list(rejd)}  (mean {np.mean(s_rej):.2f}, capped@{max(s_rej) if s_rej else 0})")
