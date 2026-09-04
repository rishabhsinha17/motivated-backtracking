"""Join the authors' own Sonnet CoT-statement labels (plot_cot_categories_v2_cache) to every
release rollout, then to our metrics. Zero API cost: labels are cached, keyed by
sha256(rendered judge prompt) with numbers blurred to X (verified against the code repo).

Outputs analysis/covertness_rollouts.jsonl (one row per joined rollout: model, question, cond,
threshold, category, last trajectory estimate, its side) and prints join rates.
"""
import glob, hashlib, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from exp2_resample import parse_final

ROOT = os.path.join(os.path.dirname(__file__), "..")
VL = "/private/tmp/claude-502/-Users-a12345-open-source-contribution/89891f11-7a46-4903-9e2e-506b1ed50e15/scratchpad/vl_code"
CACHE = os.path.join(ROOT, "tfai_data", "final_data", "cache")
LBL_V2 = os.path.join(ROOT, "tfai_data", "final_data", "plot_cot_categories_v2_cache", "main_experiment_accurate", "sonnet_extract_statement")
TRAJ = os.path.join(ROOT, "tfai_data", "final_data", "trajectories")
QUESTIONS = ["giraffes", "zills", "bridge", "crochet", "tbc", "maiden", "turns", "orangecars", "windowdays"]
ANS_RE = re.compile(r"<answer>\s*(INFLUENCED|MENTIONED|NOT_INFLUENCED|NO_STATEMENT|NO_MENTION)\s*</answer>", re.I)
NUM_RE = re.compile(r"\d[\d,\.]*")

def blur(text):
    return NUM_RE.sub("X", text)

def load_template():
    sys.path.insert(0, VL)
    from donation_bet.cot_categories_common import statement_prompt
    return statement_prompt("reasoning")

def load_labels():
    lbl = {}
    for f in glob.glob(os.path.join(LBL_V2, "*", "*.jsonl")):
        for line in open(f):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = ANS_RE.search(j.get("answer") or "")
            if m and "prompt_hash" in j:
                lbl[j["prompt_hash"]] = m.group(1).upper()
    return lbl

def load_traj_index():
    idx = {}
    for f in glob.glob(os.path.join(TRAJ, "*.jsonl")):
        for line in open(f):
            j = json.loads(line)
            a = j.get("answer")
            if a and a != "NONE":
                try:
                    idx[j["r_hash"]] = [float(x) for x in str(a).split(",") if x.strip()]
                except ValueError:
                    pass
    return idx

def main():
    tpl = load_template()
    labels = load_labels()
    tidx = load_traj_index()
    print(f"labels loaded: {len(labels)}  traj hashes: {len(tidx)}")
    out = open(os.path.join(ROOT, "analysis", "covertness_rollouts.jsonl"), "w")
    per_model = defaultdict(lambda: [0, 0])
    for model in sorted(os.listdir(CACHE)):
        for q in QUESTIONS:
            mdir = os.path.join(CACHE, model, f"v1_{q}_accurate")
            if not os.path.isdir(mdir):
                continue
            for cond in ["baseline", "below_good", "above_good"]:
                for f in glob.glob(os.path.join(mdir, f"{cond}_*.jsonl")):
                    for line in open(f).read().splitlines()[1:]:
                        j = json.loads(line)
                        r = j.get("reasoning") or ""
                        if not r.strip():
                            continue
                        h = hashlib.sha256(tpl.format(llm_text=blur(r)).encode("utf-8")).hexdigest()
                        cat = labels.get(h)
                        per_model[model][1] += 1
                        if cat is None:
                            continue
                        per_model[model][0] += 1
                        rh = hashlib.sha256(r.encode()).hexdigest()[:12]
                        traj = tidx.get(rh)
                        thr = j.get("threshold")
                        fin = parse_final(j.get("answer") or "")
                        rec = {"model": model, "q": q, "cond": cond, "thr": thr, "cat": cat,
                               "last": traj[-1] if traj else None, "n_est": len(traj) if traj else 0,
                               "final": fin}
                        out.write(json.dumps(rec) + "\n")
    out.close()
    print("join rate per model (labeled/total reasoning rows):")
    for m, (k, n) in sorted(per_model.items()):
        print(f"  {m:40s} {k}/{n}  ({k/max(1,n):.2f})")
    print("COVERTNESS_JOIN_DONE")

if __name__ == "__main__":
    main()
