"""Validate the STRONG backtracking regex against an LLM judge on a stratified sentence sample."""
import json, os, sys, re, random, time, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS, split_sentences, STRONG
ROOT = os.path.join(os.path.dirname(__file__), "..")
for line in open(os.path.join(ROOT, ".env")):
    k,_,v = line.strip().partition("=")
    if v: os.environ.setdefault(k, v)
random.seed(13)
PROMPT = """A model is doing a Fermi estimate. Here is one sentence from its reasoning:

<sentence>{s}</sentence>

Is this sentence backtracking/second-guessing (revisiting, doubting, re-checking, or revising a previous estimate or assumption — e.g. "wait", "that seems too high", "let me reconsider")? Answer only <bt>yes</bt> or <bt>no</bt>."""
TAG = re.compile(r"<bt>\s*(yes|no)\s*</bt>", re.I)
def call(prompt, tries=6):
    body = {"model":"claude-sonnet-5","max_tokens":10,
            "messages":[{"role":"user","content":prompt}]}
    for a in range(tries):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode(),
                headers={"x-api-key":os.environ["ANTHROPIC_API_KEY"],"anthropic-version":"2023-06-01","content-type":"application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d=json.loads(r.read())
            return d["content"][0]["text"]
        except Exception:
            if a==tries-1: raise
            time.sleep(min(30,2**a))
pos, neg = [], []
for m in ["glm-5p2_20260815_030703","qwen3.5-122b-a10b_20260815_030702","deepseek-v4-flash-0731_20260815_030703"]:
    rows = json.load(open(f"{RUNS}/{m}/below_good.json"))["rows"][:40]
    for r in rows:
        for s in split_sentences(r.get("reasoning") or ""):
            if len(s) < 15: continue
            (pos if STRONG.search(s) else neg).append(s)
sample = [(s,1) for s in random.sample(pos,150)] + [(s,0) for s in random.sample(neg,150)]
random.shuffle(sample)
import threading, queue
jobs = queue.Queue(); [jobs.put(x) for x in sample]
res = []; lock = threading.Lock()
def w():
    while True:
        try: s, flag = jobs.get_nowait()
        except queue.Empty: return
        try:
            v = TAG.search(call(PROMPT.format(s=s[:400])))
            if v:
                with lock: res.append((flag, v.group(1).lower()=="yes"))
        except Exception as e: print("fail", repr(e))
ts=[threading.Thread(target=w,daemon=True) for _ in range(5)]
[t.start() for t in ts]; [t.join() for t in ts]
tp=sum(1 for f,j in res if f and j); fp=sum(1 for f,j in res if f and not j)
fn=sum(1 for f,j in res if not f and j); tn=sum(1 for f,j in res if not f and not j)
print(f"n={len(res)} | regex-flagged agreeing (precision proxy): {tp}/{tp+fp} = {tp/max(1,tp+fp):.2f}")
print(f"unflagged judged-backtrack (recall gap proxy): {fn}/{fn+tn} = {fn/max(1,fn+tn):.2f}")
json.dump({"tp":tp,"fp":fp,"fn":fn,"tn":tn}, open("analysis/label_judge_validation.json","w"))
