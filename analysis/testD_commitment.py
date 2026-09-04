"""Test D: when does the bias enter the trace? Forced answer at prefix k (Datta 2604.22266 style,
two-turn stop design, identical mechanics for both models, OpenRouter chat).

For k% of the original reasoning's sentences shown as a visible assistant turn, a second user turn
forces an immediate number. bias(k) = P(above | above_good, k) - P(above | below_good, k).

PREDICTIONS WRITTEN BEFORE RUNNING (from Test A): DeepSeek's bias grows with k (bias is built in the
reasoning); GLM's bias is present at low k already (decide early) and roughly flat.

Usage: testD_commitment.py --model glm|deepseek [--n 40]
Writes exp2_out/testD_<model>.jsonl
"""
import argparse, json, os, sys, threading, queue
sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS, split_sentences
from exp2_resample import parse_final
from exp3_confession import http
from fig_transfer_testA import fallback

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "exp2_out")
SLUG = {"glm": "z-ai/glm-5.2", "deepseek": "deepseek/deepseek-v4-flash-0731"}
RUN = {"glm": "glm-5p2_20260815_030703", "deepseek": "deepseek-v4-flash-0731_20260815_030703"}
KS = [25, 50, 75, 100]
STOP = "Stop. Give your single most accurate estimate now as a single number only, nothing else."

def forced(slug, prompt, partial):
    d = http("https://openrouter.ai/api/v1/chat/completions", os.environ["OPENROUTER_API_KEY"],
             {"model": slug, "max_tokens": 2000, "temperature": 1.0,
              "messages": [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": partial},
                           {"role": "user", "content": STOP}]})
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + " " + (m.get("reasoning") or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SLUG), required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    thr = json.load(open(os.path.join(RUNS, RUN[a.model], "threshold.json")))["threshold"]
    outfile = os.path.join(OUT, f"testD_{a.model}.jsonl")
    done = set()
    if os.path.exists(outfile):
        for line in open(outfile):
            j = json.loads(line); done.add((j["cond"], j["i"], j["k"]))
    jobs = queue.Queue()
    for cond in ["below_good", "above_good"]:
        data = json.load(open(os.path.join(RUNS, RUN[a.model], f"{cond}.json")))
        prompt, rows = data["prompt"], data["rows"]
        picked = 0
        for i, row in enumerate(rows):
            r = row.get("reasoning") or ""
            if len(r) < 400: continue
            picked += 1
            if picked > a.n: break
            sents = split_sentences(r)
            for k in KS:
                if (cond, i, k) in done: continue
                cut = max(1, round(len(sents) * k / 100))
                jobs.put((cond, i, k, prompt, " ".join(sents[:cut])))
    print(f"[testD {a.model}] jobs={jobs.qsize()} thr={thr:,}")
    lock = threading.Lock()
    def worker():
        while True:
            try: cond, i, k, prompt, partial = jobs.get_nowait()
            except queue.Empty: return
            try:
                txt = forced(SLUG[a.model], prompt, partial)
                fin = parse_final(txt)
                if fin is None: fin = fallback(txt)
                rec = {"cond": cond, "i": i, "k": k, "final": fin,
                       "above": (fin or 0) > thr, "tie": fin == thr if fin is not None else None,
                       "tail": txt[-150:]}
                with lock: open(outfile, "a").write(json.dumps(rec) + "\n")
            except Exception as e:
                print("fail", cond, i, k, repr(e))
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rows = [json.loads(l) for l in open(outfile)]
    for k in KS:
        line = []
        for cond in ["below_good", "above_good"]:
            v = [r["above"] for r in rows if r["cond"] == cond and r["k"] == k and r["final"] is not None]
            line.append(f"{cond[:2]} {sum(v)/max(1,len(v)):.2f} (n={len(v)})")
        pb = [r["above"] for r in rows if r["cond"] == "below_good" and r["k"] == k and r["final"] is not None]
        pa = [r["above"] for r in rows if r["cond"] == "above_good" and r["k"] == k and r["final"] is not None]
        b = (sum(pa)/max(1,len(pa)) - sum(pb)/max(1,len(pb))) if pa and pb else float("nan")
        print(f"  k={k:3d}%  " + "  ".join(line) + f"  bias {b:+.2f}")

if __name__ == "__main__":
    main()
