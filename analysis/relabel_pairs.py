"""Judge relabel of every judge-adjacent estimate pair (closes limitation 1).

Each pair's spanning text is labeled REVISION / SCENARIO / RESTATEMENT by DeepSeek-V4-Flash
(cheap bulk judge, OpenRouter), resumable. A 150-pair random subsample is escalated to
claude-sonnet-5 for agreement (Trust-or-Escalate style). Output: exp2_out/pair_labels.jsonl.

Then rerun the R2 contrasts on the REVISION-only subset (like the "if" filter, but judged).
"""
import argparse, json, os, sys, threading, queue, random, urllib.request, time, re
sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS, split_sentences, align_trajectory

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "exp2_out", "pair_labels.jsonl")
for line in open(os.path.join(ROOT, ".env")):
    k, _, v = line.strip().partition("=")
    if v: os.environ.setdefault(k, v)

PROMPT = """A reasoning trace about a numeric estimation problem contains two consecutive candidate estimates, {v1:,.0f} then {v2:,.0f}. Here is the span of the trace from the first to the second:

<span>
{span}
</span>

Classify what the second number is, relative to the first. Exactly one label:
- REVISION: the trace treats the first number as its working estimate and the second as an update, correction or replacement of it (the running estimate actually changed).
- SCENARIO: the two numbers are alternative hypothetical cases being enumerated (e.g. "if X then A, if Y then B"), neither adopted as the working estimate at that point.
- RESTATEMENT: the second number is the same quantity restated, an intermediate arithmetic step toward it, or a different quantity entirely (not a candidate answer).

Answer with just the label."""

LBL = re.compile(r"\b(REVISION|SCENARIO|RESTATEMENT)\b", re.I)

def call(model, prompt, tries=5):
    for a in range(tries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps({"model": model, "max_tokens": 900, "temperature": 0,
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())["choices"][0]["message"]
                return (d.get("content") or "") + " " + (d.get("reasoning") or "")
        except Exception:
            if a == tries - 1: raise
            time.sleep(2 ** a)

def anthropic_call(prompt, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps({"model": "claude-sonnet-5", "max_tokens": 500,
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                         "anthropic-workspace-id": os.environ.get("ANTHROPIC_WORKSPACE_ID", ""),
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
                return " ".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        except Exception:
            if a == tries - 1: raise
            time.sleep(3 * 2 ** a)

def collect_pairs():
    pairs = []
    for d in sorted(os.listdir(RUNS)):
        rd = os.path.join(RUNS, d)
        if not os.path.exists(os.path.join(rd, "threshold.json")): continue
        thr = json.load(open(os.path.join(rd, "threshold.json")))["threshold"]
        trajs = json.load(open(os.path.join(rd, "trajectories.json")))
        for cond in ["below_good", "above_good"]:
            rows = json.load(open(os.path.join(rd, f"{cond}.json")))["rows"]
            for ri, (row, traj) in enumerate(zip(rows, trajs[cond])):
                r = row.get("reasoning") or ""
                if not r or not traj: continue
                sents = split_sentences(r)
                hits, _ = align_trajectory(sents, traj, thr)
                for (s1, o1, v1, t1), (s2, o2, v2, t2) in zip(hits, hits[1:]):
                    if t2 != t1 + 1 or v1 == v2: continue
                    span = " ".join(sents[s1:s2 + 1])[:2400]
                    pairs.append({"id": f"{d}|{cond}|{ri}|{t1}", "model": d, "cond": cond,
                                  "v1": v1, "v2": v2, "thr": thr, "span": span})
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--escalate", type=int, default=150)
    a = ap.parse_args()
    pairs = collect_pairs()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(json.loads(line)["id"])
    todo = [p for p in pairs if p["id"] not in done]
    print(f"pairs total {len(pairs)}, todo {len(todo)}")
    jobs = queue.Queue()
    for p in todo: jobs.put(p)
    lock = threading.Lock()
    def worker():
        while True:
            try: p = jobs.get_nowait()
            except queue.Empty: return
            try:
                txt = call("deepseek/deepseek-v4-flash-0731", PROMPT.format(v1=p["v1"], v2=p["v2"], span=p["span"]))
                m = LBL.search(txt)
                rec = {**{k: p[k] for k in ["id", "model", "cond", "v1", "v2", "thr"]},
                       "label": m.group(1).upper() if m else None}
                with lock: open(OUT, "a").write(json.dumps(rec) + "\n")
            except Exception as e:
                print("fail", p["id"], repr(e))
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rows = [json.loads(l) for l in open(OUT)]
    from collections import Counter
    print("labels:", Counter(r["label"] for r in rows))
    # escalate a seeded random subsample to sonnet for agreement
    random.seed(41)
    by_id = {p["id"]: p for p in pairs}
    sample = random.sample([r for r in rows if r["label"]], min(a.escalate, len(rows)))
    agree = n = 0
    esc = []
    for r in sample:
        p = by_id.get(r["id"])
        if not p: continue
        txt = anthropic_call(PROMPT.format(v1=p["v1"], v2=p["v2"], span=p["span"]))
        m = LBL.search(txt or "")
        if m:
            n += 1; agree += m.group(1).upper() == r["label"]
            esc.append({"id": r["id"], "ds": r["label"], "sonnet": m.group(1).upper()})
    json.dump(esc, open(os.path.join(ROOT, "exp2_out", "pair_labels_escalation.json"), "w"), indent=1)
    print(f"sonnet agreement on {n} escalated pairs: {agree}/{n} = {agree/max(1,n):.2f}")
    print("RELABEL_DONE")

if __name__ == "__main__":
    main()
