"""Figures for Experiment 1: motivated revision-direction forest plot + mechanism bars."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figs")
RUNS = os.path.join(HERE, "..", "value-leakage", "runs")
r1b = json.load(open(os.path.join(HERE, "exp1b_results.json")))
r1c = json.load(open(os.path.join(HERE, "exp1c_results.json")))

def mrf(model):
    f = json.load(open(os.path.join(RUNS, model, "factor.json")))
    return f["motivated_reasoning_factor"]

models = sorted(r1b.keys(), key=mrf, reverse=True)
short = {m: m.split("_")[0] for m in models}

# ---------------- Fig 1: forest plot ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
titles = [
    ("d_down_above_bg_minus_ag", "P(down|above) bg-ag",
     "Above threshold:\nΔ P(revise down)  below_good − above_good"),
    ("d_up_below_ag_minus_bg", "P(up|below)  ag-bg",
     "Below threshold:\nΔ P(revise up)  above_good − below_good"),
]
for ax, (k1b, k1c, title) in zip(axes, titles):
    ys = np.arange(len(models))[::-1]
    for y, m in zip(ys, models):
        raw = r1b[m].get(k1b)
        strat = (r1c.get(m, {}).get(k1c) or {}).get("dist")
        if raw:
            d, lo, hi = raw
            sig = lo > 0 or hi < 0
            ax.errorbar(d, y + 0.15, xerr=[[d - lo], [hi - d]], fmt="o",
                        color="#c0392b" if sig else "#7f8c8d", ms=5, capsize=2)
        if strat:
            d, lo, hi = strat
            ax.errorbar(d, y - 0.15, xerr=[[d - lo], [hi - d]], fmt="s",
                        color="#2c3e50", mfc="white", ms=5, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{short[m]}  (MRF {mrf(m):+.3f})" for m in models], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Δ probability (motivated direction > 0)", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
from matplotlib.lines import Line2D
axes[0].legend(handles=[
    Line2D([], [], marker="o", color="#c0392b", ls="", label="raw (95% cluster bootstrap CI)"),
    Line2D([], [], marker="s", color="#2c3e50", mfc="white", ls="", label="distance-stratified"),
], fontsize=8, loc="lower right")
fig.suptitle("Motivated revision-direction: given a revision, models steer toward the donation-favorable side\n"
             "(same numeric state, flipped valence; 100 rollouts/condition/model; baseline mean-reversion differenced out by the cross-condition contrast)",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIGS, "fig1_forest.png"), dpi=180)
print("wrote fig1_forest.png")

# ---------------- Fig 2: mechanism bars (headline model) ----------------
m = "qwen3.5-122b-a10b_20260815_030702"
d = r1b[m]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
conds = ["baseline", "below_good", "above_good"]
labels = ["baseline\n(no bet)", "below_good\n(above = BAD)", "above_good\n(above = GOOD)"]
colors = ["#7f8c8d", "#c0392b", "#27ae60"]
axes[0].bar(labels, [d[c]["p_down_above"] for c in conds], color=colors)
axes[0].set_title("Currently ABOVE threshold:\nP(next revision goes DOWN)", fontsize=10)
labels2 = ["baseline\n(no bet)", "below_good\n(below = GOOD)", "above_good\n(below = BAD)"]
axes[1].bar(labels2, [d[c]["p_up_below"] for c in conds], color=["#7f8c8d", "#27ae60", "#c0392b"])
axes[1].set_title("Currently BELOW threshold:\nP(next revision goes UP)", fontsize=10)
for ax in axes:
    ax.set_ylim(0, 0.85)
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                    ha="center", fontsize=8)
fig.suptitle(f"Qwen3.5-122B-A10B (threshold {json.load(open(os.path.join(RUNS, m, 'threshold.json')))['threshold']:,}): "
             "revisions steer toward whichever side the bet favors", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(FIGS, "fig2_mechanism_qwen.png"), dpi=180)
print("wrote fig2_mechanism_qwen.png")

# ---------------- cross-model: effect vs MRF ----------------
xs, ys_ = [], []
for m in models:
    a = r1b[m].get("d_down_above_bg_minus_ag"); b = r1b[m].get("d_up_below_ag_minus_bg")
    if a and b:
        xs.append(mrf(m)); ys_.append((a[0] + b[0]) / 2)
from scipy.stats import spearmanr
rho, p = spearmanr(xs, ys_)
print(f"Spearman(mean direction effect, MRF) over {len(xs)} models: rho={rho:.2f} p={p:.2f}")
