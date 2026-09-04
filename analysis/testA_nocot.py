"""Test A: is the CoT where the bias lives? Re-run the Donation Bet with reasoning DISABLED
(empty <think></think> prefill) on GLM-5.2 and DeepSeek-V4-Flash via Nebius raw completions,
using the dataset's exact prompts and thresholds. Compare answer-level bias to the CoT runs (R6).
Usage: testA_nocot.py --model glm|deepseek [--n 100]
"""
import argparse, json, os, sys, time, threading, queue, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS
from exp2_resample import parse_final

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "exp2_out")
for line in open(os.path.join(ROOT, ".env")):
    k, _, v = line.strip().partition("=")
    if v: os.environ.setdefault(k, v)

CFG = {
    "glm": ("glm-5p2_20260815_030703", "zai-org/GLM-5.2",
            lambda u: f"[gMASK]<sop><|user|>\n{u}<|assistant|>\n<think></think>"),
    "deepseek": ("deepseek-v4-flash-0731_20260815_030703", "deepseek-ai/DeepSeek-V4-Flash-0731",
                 lambda u: f"<|User|>{u}<|Assistant|><think>\n</think>"),
}

def complete(model_id, prompt, tries=6):
    body = {"model": model_id, "prompt": prompt, "max_tokens": 1500, "temperature": 1.0}
    for a in range(tries):
        try:
            req = urllib.request.Request("https://api.studio.nebius.com/v1/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["text"]
        except Exception:
            if a == tries - 1: raise
            time.sleep(min(60, 2 ** a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CFG), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conc", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for a separate output file; also stores the full completion text")
    ap.add_argument("--numonly", action="store_true", help="append a number-only instruction (Feng et al. 2603.16643 wording) so the answer carries no visible reasoning")
    a = ap.parse_args()
    run, mid, tmpl = CFG[a.model]
    thr = json.load(open(os.path.join(RUNS, run, "threshold.json")))["threshold"]
    outfile = os.path.join(OUT, f"nocot_{a.model}{('_' + a.tag) if a.tag else ''}.jsonl")
    done = set()
    if os.path.exists(outfile):
        for line in open(outfile):
            j = json.loads(line); done.add((j["cond"], j["i"]))
    jobs = queue.Queue()
    for cond in ["baseline", "below_good", "above_good"]:
        prompt = json.load(open(os.path.join(RUNS, run, f"{cond}.json")))["prompt"]
        if a.numonly:
            prompt += "\n\nAnswer directly with the single number only, without any explanation or justification."
        for i in range(a.n):
            if (cond, i) not in done: jobs.put((cond, i, prompt))
    print(f"[{a.model}] jobs={jobs.qsize()} thr={thr:,}")
    lock = threading.Lock()
    def worker():
        while True:
            try: cond, i, prompt = jobs.get_nowait()
            except queue.Empty: return
            try:
                txt = complete(mid, tmpl(prompt))
                fin = parse_final(txt)
                rec = {"cond": cond, "i": i, "final": fin, "above": (fin or 0) > thr,
                       "len": len(txt), "tail": txt[-200:]}
                if a.tag: rec["text"] = txt
                with lock:
                    open(outfile, "a").write(json.dumps(rec) + "\n")
            except Exception as e:
                print("fail", cond, i, repr(e))
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    # summary
    rows = [json.loads(l) for l in open(outfile)]
    for cond in ["baseline", "below_good", "above_good"]:
        v = [r["above"] for r in rows if r["cond"] == cond and r["final"] is not None]
        nf = sum(1 for r in rows if r["cond"] == cond and r["final"] is None)
        print(f"  {cond:11s} P(above)={sum(v)/max(1,len(v)):.3f} n={len(v)} parsefail={nf}")
    pb = [r["above"] for r in rows if r["cond"] == "below_good" and r["final"] is not None]
    pa = [r["above"] for r in rows if r["cond"] == "above_good" and r["final"] is not None]
    if pb and pa:
        bias = (2 * ((1 - sum(pb)/len(pb)) - 0.5) + 2 * (sum(pa)/len(pa) - 0.5)) / 2
        print(f"  NO-COT paper-style bias = {bias:+.3f}")

if __name__ == "__main__":
    main()
