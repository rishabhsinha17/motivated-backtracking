"""Figure 9: the report channels. (a) verbal admission: prospective (C2), retrospective neutral (Exp 3), third person (Exp 3).
(b) numeric bias: actual CoT, self forecast (C), analyst frame (B), no CoT (A pooled)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, min(c - h, p), max(c + h, p)

def confess(m, cond, arm):
    f = os.path.join(ROOT, "exp2_out", f"confess_{m}_{cond}.jsonl" if m != "deepseek" else f"confess_deepseek_or_{cond}.jsonl")
    rows = [json.loads(l) for l in open(f)]
    v = [r["verdict"] for r in rows if r["arm"] == arm and r["verdict"]]
    return wilson(sum(x == "yes" for x in v), len(v))

def prospective(m, cond):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "exp2_out", f"testC2_{m}.jsonl"))]
    v = [r["verdict"] for r in rows if r["cond"] == cond and r["verdict"]]
    return wilson(sum(x == "yes" for x in v), len(v))

bc = json.load(open(os.path.join(ROOT, "analysis", "testBC_results.json")))
ta = json.load(open(os.path.join(ROOT, "analysis", "testA_results.json")))
models = [("glm", "GLM-5.2"), ("deepseek", "DeepSeek-V4-Flash")]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
# (a) verbal channels
ax = axes[0]
labels = ["prospective\n(before acting)", "retrospective\nneutral", "retrospective\nthird person"]
w = 0.18
for mi, (m, name) in enumerate(models):
    for ci_, cond in enumerate(["below_good", "above_good"]):
        vals = [prospective(m, cond), confess(m, cond, "N"), confess(m, cond, "P")]
        xs = np.arange(3) + (mi * 2 + ci_ - 1.5) * w
        ax.bar(xs, [v[0] for v in vals], width=w, color=["#4c72b0", "#dd8452"][mi], alpha=[0.95, 0.55][ci_],
               label=f"{name} {cond}" )
        ax.errorbar(xs, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor="black", capsize=2, lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("P(admits bet influence)"); ax.set_ylim(0, 1.05)
ax.set_title("(a) Verbal report is a fixed per model stance", fontsize=10)
ax.legend(fontsize=6.5, loc="upper left")
# (b) numeric channels
ax = axes[1]
labels = ["actual answer\n(CoT)", "self forecast\n(C)", "analyst frame\n(B)", "no CoT\n(A)"]
for mi, (m, name) in enumerate(models):
    key_c, key_b = f"{m}_C", f"{m}_B"
    vals = [(bc[key_c]["cot_bias"], None, None),
            (bc[key_c]["bias"], *bc[key_c]["ci"]),
            (bc[key_b]["bias"], *bc[key_b]["ci"]) if key_b in bc else (np.nan, np.nan, np.nan),
            (ta[f"{m}_pooled"]["bias"], *ta[f"{m}_pooled"]["ci"])]
    xs = np.arange(4) + (mi - 0.5) * 0.36
    ax.bar(xs, [v[0] for v in vals], width=0.34, color=["#4c72b0", "#dd8452"][mi], label=name)
    for x, v in zip(xs, vals):
        if v[1] is not None and not np.isnan(v[1]):
            ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor="black", capsize=3, lw=0.8)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("bias  P(above | good above) − P(above | good below)")
ax.set_title("(b) Numeric channels move, in model specific directions", fontsize=10)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figs", "fig9_channels.png"), dpi=200)
print("wrote fig9_channels.png")
