"""Estimate-judge shim: same verbatim prompt + parser as the repo's judge.py, but
transported via OpenRouter (anthropic/claude-sonnet-5) because the user's Anthropic
key is identity-linked and needs a workspace id the repo client doesn't send.
Deviations from repo judge: transport (OpenRouter), temperature=0. Prompt/parser identical.

Usage: judge_shim.py <run_dir> [<run_dir> ...]
Writes <run_dir>/estimates.json (same shape as repo judge).
"""
import json, os, sys, time, threading, queue, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "value-leakage", "src"))
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate, CONDITIONS

ROOT = os.path.join(os.path.dirname(__file__), "..")
for line in open(os.path.join(ROOT, ".env")):
    k, _, v = line.strip().partition("=")
    if v:
        os.environ.setdefault(k, v)

def call(prompt, tries=8):
    body = {"model": "anthropic/claude-sonnet-5", "max_tokens": 300, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(tries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** attempt) if e.code == 429 else 2 ** attempt)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)

def judge_run(run_dir):
    run_dir = Path(run_dir)
    out = {}
    for cond in CONDITIONS:
        p = run_dir / f"{cond}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())["rows"]
        sources = [(r.get("content") or "").strip() or None for r in rows]
        parsed = [None] * len(sources)
        prev_file = run_dir / "estimates.json"
        if prev_file.exists():
            prev = json.loads(prev_file.read_text()).get(cond)
            if prev and len(prev) == len(sources):
                parsed = prev  # merge: keep already-parsed, refill only Nones
        jobs = queue.Queue()
        for i, s in enumerate(sources):
            if s and parsed[i] is None:
                jobs.put((i, s))
        lock = threading.Lock()
        def worker():
            while True:
                try:
                    i, s = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    resp = call(NUMBER_JUDGE_PROMPT.format(llm_text=s))
                    val = parse_tagged_estimate(resp)
                    with lock:
                        parsed[i] = val
                except Exception as e:
                    print(f"  idx {i} failed: {e!r}")
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        ok = sum(1 for x in parsed if x is not None)
        print(f"{run_dir.name} {cond}: {ok}/{len(sources)} parsed")
        out[cond] = parsed
    (run_dir / "estimates.json").write_text(json.dumps(out, indent=2))
    print(f"saved {run_dir}/estimates.json")

if __name__ == "__main__":
    for rd in sys.argv[1:]:
        judge_run(rd)
