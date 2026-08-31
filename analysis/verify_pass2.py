"""Independent second verification of the 50-pair label sheet.
Method differs from prelabeling: (a) est pair checked against FULL transcripts, not excerpts;
(b) backtrack + revision-vs-scenario judged per row by claude-sonnet-5."""
import json, os, re, sys, time, urllib.request, threading, queue
sys.path.insert(0, "analysis")
from exp1_backtracking import RUNS, numbers_in, close
for line in open(".env"):
    k,_,v = line.strip().partition("=")
    if v: os.environ.setdefault(k, v)
RUN_BY = {"qwen3.5-122b-a10b":"qwen3.5-122b-a10b_20260815_030702",
          "deepseek-v4-flash-0731":"deepseek-v4-flash-0731_20260815_030703",
          "inkling":"inkling_20260815_030703",
          "claude-opus-4-7":"claude-opus-4-7_20260815_042213"}
BT_PROMPT = """A model is doing a Fermi estimate. Here is a passage from its reasoning:

<passage>{s}</passage>

Does this passage contain backtracking/second-guessing (revisiting, doubting, re-checking, or revising a previous estimate or assumption)? Answer only <bt>yes</bt> or <bt>no</bt>."""
REV_PROMPT = """A model doing a Fermi estimate produced two consecutive candidate values: {v1} then {v2}. Here is the passage between and including them:

<passage>{s}</passage>

Is the second value a genuine REVISION (the model updating its working estimate) or SCENARIO enumeration (listing conditional alternatives like "if X then A, if Y then B" without committing)? Answer only <r>revision</r> or <r>scenario</r> or <r>unclear</r>."""
def call(prompt, tries=6):
    body = {"model":"claude-sonnet-5","max_tokens":10,"messages":[{"role":"user","content":prompt}]}
    for a in range(tries):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode(),
                headers={"x-api-key":os.environ["ANTHROPIC_API_KEY"],"anthropic-version":"2023-06-01","content-type":"application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["content"][0]["text"]
        except Exception:
            if a==tries-1: raise
            time.sleep(min(30,2**a))
rows = [l.split("\t") for l in open("verification/label_sheet_prelabeled.tsv").read().splitlines()[1:]]
cache = {}
def reasoning_for(model, cond, i):
    key = (model, cond)
    if key not in cache:
        cache[key] = json.load(open(f"value-leakage/runs/{RUN_BY[model]}/{cond}.json"))["rows"]
    return cache[key][int(i)].get("reasoning") or ""
results = []
lock = threading.Lock(); jobs = queue.Queue()
for f in rows: jobs.put(f)
def worker():
    while True:
        try: f = jobs.get_nowait()
        except queue.Empty: return
        rid, model, cond, ro, srange, v1s, v2s, thr_s = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]
        v1, v2 = float(v1s.replace(",","")), float(v2s.replace(",",""))
        thr = float(thr_s.replace(",",""))
        rs = reasoning_for(model, cond, ro)
        nums = [v for _, v in numbers_in(rs)]
        f1 = any(close(n, v1) for n in nums); f2 = any(close(n, v2) for n in nums)
        thr_collision = close(v1, thr, 0.0005) or close(v2, thr, 0.0005)
        est2 = "y" if (f1 and f2 and not thr_collision) else ("thr-collision" if thr_collision else "missing")
        excerpt = f[15][:800]
        bt2 = "y" if "<bt>yes" in call(BT_PROMPT.format(s=excerpt)).lower() else "n"
        rv = call(REV_PROMPT.format(v1=v1s, v2=v2s, s=excerpt)).lower()
        rev2 = "revision" if "revision" in rv else ("scenario" if "scenario" in rv else "unclear")
        with lock:
            results.append({"id": int(rid), "est2": est2, "bt2_says_backtrack": bt2,
                            "pipeline_bt": f[10], "rev2": rev2,
                            "pre_est": f[11], "pre_rev": f[12], "pre_btok": f[13]})
threads = [threading.Thread(target=worker, daemon=True) for _ in range(5)]
[t.start() for t in threads]; [t.join() for t in threads]
results.sort(key=lambda r: r["id"])
json.dump(results, open("verification/pass2_results.json","w"), indent=1)
# summarize
est_ok = sum(1 for r in results if r["est2"]=="y")
thr_c  = sum(1 for r in results if r["est2"]=="thr-collision")
miss   = sum(1 for r in results if r["est2"]=="missing")
bt_agree = sum(1 for r in results if (r["bt2_says_backtrack"]=="y") == (r["pipeline_bt"]=="True"))
rev_real = sum(1 for r in results if r["rev2"]=="revision")
rev_scen = sum(1 for r in results if r["rev2"]=="scenario")
print(f"EST full-transcript: {est_ok}/50 both values found, {thr_c} threshold-collision, {miss} missing")
print(f"BT flag vs judge: {bt_agree}/50 agree")
print(f"REV judge: {rev_real} revision, {rev_scen} scenario, {50-rev_real-rev_scen} unclear")
dis = [r for r in results if
       (r["est2"]=="missing" and r["pre_est"] in ("y",)) or
       (r["est2"]=="y" and r["pre_est"]=="n") or
       ((r["bt2_says_backtrack"]=="y") != (r["pipeline_bt"]=="True")) != (r["pre_btok"]=="n") or
       ((r["rev2"]=="scenario") != (r["pre_rev"]=="n") and r["rev2"]!="unclear")]
print(f"\nDISAGREEMENT ROWS between pass1 (mine) and pass2 (independent): {len(dis)}")
for r in dis: print(" ", r)
