"""Figure: commitment curve (Test D). bias(k) with Wilson-derived bootstrap CIs per model."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
rng = np.random.default_rng(0)
KS = [25, 50, 75, 100]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for m, name, col in [("glm", "GLM-5.2", "#4c72b0"), ("deepseek", "DeepSeek-V4-Flash", "#dd8452")]:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "exp2_out", f"testD_{m}.jsonl"))]
    xs, ys, lo, hi = [], [], [], []
    for k in KS:
        pb = [r["above"] for r in rows if r["cond"] == "below_good" and r["k"] == k and r["final"] is not None]
        pa = [r["above"] for r in rows if r["cond"] == "above_good" and r["k"] == k and r["final"] is not None]
        if not pb or not pa: continue
        b = np.mean(pa) - np.mean(pb)
        boots = [np.mean(rng.choice(pa, len(pa))) - np.mean(rng.choice(pb, len(pb))) for _ in range(3000)]
        xs.append(k); ys.append(b); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                fmt="o-", color=col, capsize=3, label=name)
# anchors from Test A (k=0 equivalents: number-only) and full CoT (R6)
ax.scatter([0, 0], [0.40, 0.01], marker="s", color=["#4c72b0", "#dd8452"], zorder=5)
ax.annotate("number only (Test A)", (0, 0.40), xytext=(4, 6), textcoords="offset points", fontsize=7, color="#4c72b0")
ax.annotate("number only", (0, 0.01), xytext=(4, -10), textcoords="offset points", fontsize=7, color="#dd8452")
ax.scatter([104, 104], [0.32, 0.42], marker="D", color=["#4c72b0", "#dd8452"], zorder=5)
ax.annotate("natural CoT (R6)", (104, 0.42), xytext=(-64, 6), textcoords="offset points", fontsize=7, color="#dd8452")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("% of the original reasoning shown before the forced answer")
ax.set_ylabel("bias  P(above | good above) − P(above | good below)")
ax.set_title("Test D: when the bias enters. GLM commits early, DeepSeek accumulates.", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figs", "fig12_commitment.png"), dpi=200)
print("fig12 written")
