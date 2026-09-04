"""Summarise confession-grid files: P(admit) per arm x condition with Wilson 95% CIs.
Usage: exp3_summarise.py glm deepseek qwen"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
for m in sys.argv[1:] or ["glm", "deepseek", "qwen"]:
    for cond in ["below_good", "above_good"]:
        f = os.path.join(ROOT, "exp2_out", f"confess_{m}_{cond}.jsonl")
        if not os.path.exists(f): continue
        rows = [json.loads(l) for l in open(f)]
        line = []
        for arm, name in [("N", "neutral"), ("A", "amnesty"), ("T", "threat"), ("O", "private"), ("P", "third person")]:
            v = [r["verdict"] for r in rows if r["arm"] == arm]
            ok = [x for x in v if x]
            p, lo, hi = wilson(sum(x == "yes" for x in ok), len(ok))
            line.append(f"{name} {p:.2f} [{lo:.2f},{hi:.2f}] n={len(ok)}/{len(v)}")
        print(f"{m:9s} {cond:11s} " + " | ".join(line))
