"""Transfer test: run the Exp 1 pipeline on the TruthfulAI value_leakage_data release
(9 questions x many models) by materializing run_dirs in the adsingh-64 format.

Release layout (verified):
  tfai_data/final_data/cache/<model>/<prompt_key>/{baseline,below_good,above_good}_<hash>.jsonl
     line 1 header {hash, model_name, prompt_key, direction, n_per_threshold, thresholds}
     lines 2.. rollouts {reasoning, answer, prompt, threshold}
  tfai_data/final_data/trajectories/*.jsonl rows {r_hash, answer}  (r_hash = sha256(reasoning)[:12];
     answer = "v1,v2,..." or "NONE")
Output: transfer_runs/<model>__<question>/{baseline,below_good,above_good}.json, trajectories.json,
threshold.json  -> consumable by exp1_backtracking / exp1b_revisions via RUNS override.
"""
import glob, hashlib, json, os, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "tfai_data", "final_data", "cache")
TRAJ = os.path.join(ROOT, "tfai_data", "final_data", "trajectories")
OUT = os.path.join(ROOT, "transfer_runs")
QUESTIONS = ["giraffes", "zills", "bridge", "crochet", "tbc", "maiden", "turns", "orangecars", "windowdays"]

def load_traj_index():
    idx = {}
    for f in glob.glob(os.path.join(TRAJ, "*.jsonl")):
        for line in open(f):
            j = json.loads(line)
            a = j.get("answer")
            if a is None or a == "NONE":
                idx[j["r_hash"]] = None
            else:
                try:
                    idx[j["r_hash"]] = [int(x) for x in str(a).split(",") if x.strip()]
                except ValueError:
                    idx[j["r_hash"]] = None
    return idx

def rhash(reasoning):
    return hashlib.sha256(reasoning.encode()).hexdigest()[:12]

def materialize(model, question, tidx):
    mdir = os.path.join(CACHE, model, f"v1_{question}_accurate")
    if not os.path.isdir(mdir):
        return None
    out_dir = os.path.join(OUT, f"{model}__{question}")
    os.makedirs(out_dir, exist_ok=True)
    # modal threshold across bet rows; multi-threshold files (E.9 percentile runs) get filtered to it
    thr_counts = defaultdict(int)
    for cond in ["below_good", "above_good"]:
        for f in glob.glob(os.path.join(mdir, f"{cond}_*.jsonl")):
            for line in open(f).read().splitlines()[1:]:
                t = json.loads(line).get("threshold")
                if t is not None: thr_counts[int(t)] += 1
    if not thr_counts:
        return None
    thr = max(thr_counts, key=thr_counts.get)
    trajs, stats = {}, {}
    for cond in ["baseline", "below_good", "above_good"]:
        files = glob.glob(os.path.join(mdir, f"{cond}_*.jsonl"))
        if not files:
            return None
        rows, tlist, matched = [], [], 0
        for f in files:
            lines = open(f).read().splitlines()
            for line in lines[1:]:
                j = json.loads(line)
                if cond != "baseline" and j.get("threshold") is not None and int(j["threshold"]) != thr:
                    continue
                r = j.get("reasoning") or ""
                rows.append({"reasoning": r, "content": j.get("answer") or "", "finish_reason": "stop"})
                t = tidx.get(rhash(r)) if r else None
                if t: matched += 1
                tlist.append(t)
        json.dump({"model": model, "condition": cond, "prompt": "", "rows": rows},
                  open(os.path.join(out_dir, f"{cond}.json"), "w"))
        trajs[cond] = tlist
        stats[cond] = (matched, len(rows))
    json.dump({"threshold": thr, "n_thresholds": len(thr_counts)}, open(os.path.join(out_dir, "threshold.json"), "w"))
    json.dump(trajs, open(os.path.join(out_dir, "trajectories.json"), "w"))
    return stats, thr, len(thr_counts)

if __name__ == "__main__":
    models = sys.argv[1:] or sorted(os.listdir(CACHE))
    tidx = load_traj_index()
    print(f"trajectory index: {len(tidx)} hashes")
    for m in models:
        for q in QUESTIONS:
            r = materialize(m, q, tidx)
            if r:
                stats, thr, nthr = r
                s = " ".join(f"{c[:2]}={stats[c][0]}/{stats[c][1]}" for c in stats)
                print(f"{m:32s} {q:11s} thr={thr:<14,} multi_thr={nthr}  traj_matched {s}")
