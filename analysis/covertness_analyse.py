"""Analyse the authors' Sonnet CoT-statement labels joined to release rollouts.

Q1  Is the explicit in-CoT denial faithful? P(final on good side | NOT_INFLUENCED) vs baseline 0.5
    and vs INFLUENCED, per model and pooled. Outcome = final answer side (parse of answer text,
    falling back to last judge-trajectory estimate; R7 validated last ≈ final).
Q2  Cross-model: does in-CoT admission (INFLUENCED rate) or denial (NOT_INFLUENCED rate) track
    steering strength (transfer_results) and answer bias?
Writes analysis/covertness_results.json.
"""
import json, os, collections
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "analysis", "covertness_rollouts.jsonl"))]
for r in rows:
    v = r["final"] if r["final"] is not None else r["last"]
    r["v"] = v
    r["good"] = None
    if v is not None and r["thr"] is not None:
        above = v > r["thr"]
        r["good"] = above if r["cond"] == "above_good" else (not above) if r["cond"] == "below_good" else None
rows = [r for r in rows if r["cond"] in ("below_good", "above_good")]
usable = [r for r in rows if r["good"] is not None]
print(f"incentive rows {len(rows)}, with outcome {len(usable)} ({len(usable)/len(rows):.2f})")

CATS = ["INFLUENCED", "MENTIONED", "NOT_INFLUENCED", "NO_STATEMENT", "NO_MENTION"]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), 0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, c-h, c+h

# pooled P(good | category)
print("\n== pooled P(final on good side | category) ==")
pooled = {}
for c in CATS:
    v = [r["good"] for r in usable if r["cat"] == c]
    p, lo, hi = wilson(sum(v), len(v))
    pooled[c] = {"p": p, "lo": lo, "hi": hi, "n": len(v)}
    print(f"  {c:14s} {p:.3f} [{lo:.3f},{hi:.3f}]  n={len(v)}")

# per model
bym = collections.defaultdict(list)
for r in usable: bym[r["model"]].append(r)
per_model = {}
print("\n== per model: NOT_INFLUENCED bias vs INFLUENCED bias (P(good)−0.5, ×2 = paper-style) ==")
for m, v in sorted(bym.items()):
    e = {}
    for c in CATS:
        vv = [r["good"] for r in v if r["cat"] == c]
        e[c] = wilson(sum(vv), len(vv)) + (len(vv),)
    rates = {c: sum(r["cat"] == c for r in v)/len(v) for c in CATS}
    per_model[m] = {"cat_p_good": {c: e[c][:3] for c in CATS}, "cat_n": {c: e[c][3] for c in CATS}, "rates": rates, "n": len(v)}
    ni, inf = e["NOT_INFLUENCED"], e["INFLUENCED"]
    flag = "DENIAL-BIASED" if ni[1] > 0.5 else ("" if np.isnan(ni[0]) else "denial ok")
    print(f"  {m:40s} NOT_INFL {ni[0]:.2f} [{ni[1]:.2f},{ni[2]:.2f}] n={ni[3]:4d} | INFL {inf[0]:.2f} n={inf[3]:4d} | rate(NOT_INFL)={rates['NOT_INFLUENCED']:.2f} rate(INFL)={rates['INFLUENCED']:.2f}  {flag}")

# cross-model correlations with steering + bias
tr = json.load(open(os.path.join(ROOT, "analysis", "transfer_results.json")))
steer = collections.defaultdict(list)
for t in tr: steer[t["model"]].append(t["d"])
steer = {m: float(np.mean(v)) for m, v in steer.items()}
bias_m = {}
for m, v in bym.items():
    pa = [r["good"] for r in v if r["cond"] == "above_good"]
    pb = [r["good"] for r in v if r["cond"] == "below_good"]
    if len(pa) > 30 and len(pb) > 30:
        # good means above in above_good and below in below_good; bias = P(above|ag) - P(above|bg)
        bias_m[m] = float(np.mean(pa) - (1 - np.mean(pb)))
common = [m for m in per_model if m in steer]
print(f"\n== cross-model (n={len(common)} models with steering data) ==")
for feat in ["INFLUENCED", "NOT_INFLUENCED", "NO_MENTION"]:
    x = [per_model[m]["rates"][feat] for m in common]
    for name, y in (("steering", [steer[m] for m in common]), ("answer bias", [bias_m.get(m, np.nan) for m in common])):
        mask = ~np.isnan(y)
        rho, p = spearmanr(np.array(x)[mask], np.array(y)[mask])
        print(f"  rate({feat}) vs {name:11s}: rho={rho:+.2f} p={p:.3f}")

json.dump({"pooled": pooled, "per_model": per_model, "steer": steer, "bias": bias_m},
          open(os.path.join(ROOT, "analysis", "covertness_results.json"), "w"), indent=1, default=float)
print("\nCOVERTNESS_ANALYSE_DONE")
