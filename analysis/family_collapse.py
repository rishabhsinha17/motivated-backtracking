"""Ibragimov-Muller family-level inference for the headline counts (methodology sweep item 1).

Model variants within a family share training lineage, so cell counts overstate independence.
Collapse each contrast set to one mean per family, then a one-sample t-test across families.
Applies to: transfer revision direction (R8), stopping (R3b), bunching (R8c), denial bias (R8b).
"""
import json, os, re, collections
import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")

def family(m):
    m = m.lower()
    for f in ["claude", "gpt", "gemini", "qwen", "kimi", "deepseek", "glm", "inkling", "minimax"]:
        if f in m: return f
    return m.split("-")[0]

def im_test(values_by_family, label):
    fams = {f: np.mean(v) for f, v in values_by_family.items() if len(v) > 0}
    x = np.array(list(fams.values()))
    t, p = stats.ttest_1samp(x, 0)
    p_one = p / 2 if t > 0 else 1 - p / 2
    print(f"{label:34s} q={len(x)} families, mean {x.mean():+.3f}, t({len(x)-1})={t:.2f}, one-sided p={p_one:.4g}")
    print(f"{'':34s} family means: " + " ".join(f"{f}={v:+.3f}" for f, v in sorted(fams.items())))
    return {"q": len(x), "mean": float(x.mean()), "t": float(t), "p_one_sided": float(p_one),
            "families": {f: float(v) for f, v in fams.items()}}

out = {}
tr = json.load(open(os.path.join(ROOT, "analysis", "transfer_results.json")))
by = collections.defaultdict(list)
for r in tr: by[family(r["model"])].append(r["d"])
out["transfer_direction"] = im_test(by, "R8 transfer revision direction")

sp = json.load(open(os.path.join(ROOT, "analysis", "stopping_results.json")))
by = collections.defaultdict(list)
for r in sp["release"]: by[family(r["model"].split("__")[0])].append(r["d"])
for r in sp["dataset"]: by[family(r["model"])].append(r["d"])
out["stopping"] = im_test(by, "R3b motivated stopping")

bu = json.load(open(os.path.join(ROOT, "analysis", "bunching_results.json")))
by = collections.defaultdict(list)
for m, v in bu.items(): by[family(m)].append(v["b"])
out["bunching"] = im_test(by, "R8c bunching")

cov = json.load(open(os.path.join(ROOT, "analysis", "covertness_results.json")))
by = collections.defaultdict(list)
for m, v in cov["per_model"].items():
    p = v["cat_p_good"]["NOT_INFLUENCED"][0]
    if not np.isnan(p) and v["cat_n"]["NOT_INFLUENCED"] >= 50:
        by[family(m)].append(p - 0.5)
out["denial_bias"] = im_test(by, "R8b denial-labeled bias")

e1b = json.load(open(os.path.join(ROOT, "analysis", "exp1b_results.json")))
by = collections.defaultdict(list)
for k, v in e1b.items():
    bg, ag = v["below_good"], v["above_good"]
    steer = ((bg["p_down_above"] - ag["p_down_above"]) + (ag["p_up_below"] - bg["p_up_below"])) / 2
    by[family(k)].append(steer)
out["dataset_steering"] = im_test(by, "R2 dataset revision direction")

json.dump(out, open(os.path.join(ROOT, "analysis", "family_collapse.json"), "w"), indent=1)
print("FAMILY_COLLAPSE_DONE")
