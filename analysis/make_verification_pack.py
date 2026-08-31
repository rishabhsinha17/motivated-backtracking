"""Builds the human-verification pack:
  verification/annotated_<model>_<cond>_<i>.txt  - full CoT with inline pipeline annotations
  verification/label_sheet.tsv                   - 50 random revision-pairs to hand-label
Deterministic (seed 7). These are RANDOM samples, not cherry-picked.
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS, split_sentences, align_trajectory, STRONG
from exp1b_revisions import thr_regexes

RNG = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(__file__), "..", "verification")
os.makedirs(OUT, exist_ok=True)

PICK_MODELS = ["qwen3.5-122b-a10b_20260815_030702", "claude-opus-4-7_20260815_042213",
               "inkling_20260815_030703", "deepseek-v4-flash-0731_20260815_030703"]
CONDS = ["below_good", "above_good"]

pairs_pool = []
for m in PICK_MODELS:
    rd = os.path.join(RUNS, m)
    thr = json.load(open(os.path.join(rd, "threshold.json")))["threshold"]
    trajs = json.load(open(os.path.join(rd, "trajectories.json")))
    for cond in CONDS:
        rows = json.load(open(os.path.join(rd, f"{cond}.json")))["rows"]
        idxs = [i for i, (r, t) in enumerate(zip(rows, trajs[cond]))
                if (r.get("reasoning") and t)]
        chosen = RNG.choice(idxs, size=min(3, len(idxs)), replace=False)
        for i in chosen:
            reasoning, traj = rows[i]["reasoning"], trajs[cond][i]
            sentences = split_sentences(reasoning)
            hits, mr = align_trajectory(sentences, traj, thr)
            est_by_sent = {}
            for si, off, val, _tj in hits:
                est_by_sent.setdefault(si, []).append(val)
            lines = [f"MODEL {m}  COND {cond}  ROLLOUT {i}  THRESHOLD {thr:,}",
                     f"judge trajectory ({len(traj)}): {traj}",
                     f"alignment match rate: {mr:.2f}", "-" * 80]
            for si, s in enumerate(sentences):
                tags = []
                for v in est_by_sent.get(si, []):
                    tags.append(f"EST {v:,.0f} [{'ABOVE' if v > thr else 'BELOW'}]")
                if STRONG.search(s):
                    tags.append("BACKTRACK")
                tag = ("  <<< " + "; ".join(tags)) if tags else ""
                lines.append(f"[{si:03d}] {s}{tag}")
            fn = os.path.join(OUT, f"annotated_{m.split('_')[0]}_{cond}_{i}.txt")
            open(fn, "w").write("\n".join(lines))
            # collect revision pairs for the label sheet
            for (s1, o1, v1, t1), (s2, o2, v2, t2) in zip(hits, hits[1:]):
                if v1 == v2 or t2 != t1 + 1:
                    continue
                excerpt = " ".join(sentences[max(0, s1):s2 + 1])[:600]
                pairs_pool.append({
                    "model": m.split("_")[0], "cond": cond, "rollout": int(i),
                    "sent_range": f"{s1}-{s2}", "v1": v1, "v2": v2, "thr": thr,
                    "pipeline_side": "above" if v1 > thr else "below",
                    "pipeline_direction": "down" if v2 < v1 else "up",
                    "pipeline_backtrack": bool(STRONG.search(excerpt)),
                    "excerpt": excerpt.replace("\t", " ").replace("\n", " "),
                })

sel = RNG.choice(len(pairs_pool), size=min(50, len(pairs_pool)), replace=False)
hdr = ["id", "model", "cond", "rollout", "sent_range", "v1", "v2", "threshold",
       "pipeline_side", "pipeline_direction", "pipeline_backtrack",
       "HUMAN_est_pair_correct(y/n)", "HUMAN_is_real_revision(y/n)",
       "HUMAN_backtrack_flag_correct(y/n)", "HUMAN_notes", "excerpt"]
with open(os.path.join(OUT, "label_sheet.tsv"), "w") as f:
    f.write("\t".join(hdr) + "\n")
    for j, k in enumerate(sel):
        p = pairs_pool[k]
        f.write("\t".join(str(x) for x in [
            j, p["model"], p["cond"], p["rollout"], p["sent_range"],
            f"{p['v1']:,.0f}", f"{p['v2']:,.0f}", f"{p['thr']:,}",
            p["pipeline_side"], p["pipeline_direction"], p["pipeline_backtrack"],
            "", "", "", "", p["excerpt"]]) + "\n")
print(f"wrote {len(os.listdir(OUT))-1} annotated transcripts + label_sheet.tsv ({len(sel)} rows)")
