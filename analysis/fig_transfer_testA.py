"""Figures for the transfer test (fig7) and the no-CoT prediction test (fig8)."""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

# ---------- fig7: transfer heatmap (model x question, pooled above/below contrast) ----------
rows = json.load(open(os.path.join(ROOT, "analysis", "transfer_results.json")))
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["model"], r["question"])].append(r["d"])
models = sorted({m for m, _ in cells}, key=lambda m: -np.mean([np.mean(v) for (mm, _), v in cells.items() if mm == m]))
questions = ["giraffes", "zills", "bridge", "crochet", "tbc", "maiden", "turns", "orangecars", "windowdays"]
M = np.full((len(models), len(questions)), np.nan)
for i, m in enumerate(models):
    for j, q in enumerate(questions):
        if (m, q) in cells:
            M[i, j] = np.mean(cells[(m, q)])
fig, ax = plt.subplots(figsize=(9, 0.32 * len(models) + 1.6))
lim = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(questions))); ax.set_xticklabels(questions, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=7)
for i in range(len(models)):
    for j in range(len(questions)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(M[i, j]) > 0.6 * lim else "black")
cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("mean revision-direction contrast (bad-side minus good-side, P(revise toward good side))", fontsize=7)
pos = sum(r["d"] > 0 for r in rows); sig = sum(r["lo"] > 0 for r in rows)
ax.set_title(f"Transfer: revision steering on the authors' release. {pos}/{len(rows)} contrasts motivated, {sig} significant, "
             f"{sum(r['hi']<0 for r in rows)} reversal", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7_transfer.png"), dpi=200)
plt.close()

# ---------- fig8: no-CoT vs CoT bias ----------
r7 = json.load(open(os.path.join(ROOT, "analysis", "r7_answer_bias.json")))
import re
MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "trillion": 1e12, "t": 1e12, "quadrillion": 1e15}
NUM = re.compile(r"(\d[\d,]*\.?\d*(?:e[+-]?\d+)?)\s*(thousand|million|billion|trillion|quadrillion|k|m|b|t)?\b", re.I)
THR = {"glm": 20874000, "deepseek": 23700000}
def fallback(txt):
    best = None
    for mm in NUM.finditer(txt[-400:]):
        try: v = float(mm.group(1).replace(",", ""))
        except ValueError: continue
        if mm.group(2): v *= MULT[mm.group(2).lower()]
        if v >= 1000: best = v
    return best
def nocot(m, files=None):
    rr = []
    for f in (files or (f"nocot_{m}.jsonl", f"nocot_{m}_full.jsonl")):
        p = os.path.join(ROOT, "exp2_out", f)
        if os.path.exists(p): rr += [json.loads(l) for l in open(p)]
    for r in rr:
        if r["final"] is None:
            v = fallback(r.get("text") or r["tail"])
            if v is not None: r["final"], r["above"] = v, v > THR[m]
    return {c: [r["above"] for r in rr if r["cond"] == c and r["final"] is not None] for c in ["baseline", "below_good", "above_good"]}
def cot(key):
    e = r7[key]
    return {c: [True] * round(e[c]["p"] * e[c]["n"]) + [False] * (e[c]["n"] - round(e[c]["p"] * e[c]["n"])) for c in ["baseline", "below_good", "above_good"]}
def bias(d): return np.mean(d["above_good"]) - np.mean(d["below_good"])
def boot_ci(d, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        dd = {c: rng.choice(d[c], len(d[c])) for c in d}
        out.append(bias(dd))
    return np.percentile(out, [2.5, 97.5])
pairs = [("GLM-5.2", "glm", "glm-5p2_20260815_030703"), ("DeepSeek-V4-Flash", "deepseek", "deepseek-v4-flash-0731_20260815_030703")]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, (name, m, key) in zip(axes, pairs):
    dnum, dn, dc = nocot(m, [f"nocot_{m}_numonly.jsonl"]), nocot(m), cot(key)
    vals = [bias(dnum), bias(dn), bias(dc)]
    cis = [boot_ci(dnum), boot_ci(dn), boot_ci(dc)]
    xs = [0, 1, 2]
    ax.bar(xs, vals, color=["#8c8c8c", "#bdbdbd", "#c44e52"], width=0.55)
    for x, v, ci in zip(xs, vals, cis):
        ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="black", capsize=4)
        ax.text(x, ci[1] + 0.02, f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(["number only\n(no reasoning)", "thinking off\n(short justification)", "full CoT"], fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(name, fontsize=10)
    ax.set_ylim(-0.1, 0.65)
axes[0].set_ylabel("answer bias\nP(above | good above) − P(above | good below)", fontsize=8)
fig.suptitle("Test A: whether the bias needs reasoning is model specific (per condition: number only n≈100, thinking off n≈190, full CoT n=100)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig8_nocot.png"), dpi=200)
print("wrote fig7_transfer.png, fig8_nocot.png")
