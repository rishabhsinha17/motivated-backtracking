"""Experiment 2: on-policy causal resampling for motivated backtracking.

Arms (per rollout, at one fork point each):
  A  bad-side backtrack suppression: at the first STRONG-backtrack sentence that follows a
     bad-side estimate, reject-and-resample until the continuation's first sentence is not a
     backtrack (K tries), then roll to completion.  Rejected-count = resilience statistic.
  B  good-side control: same at the first good-side backtrack sentence.
  C  plain resample at the same fork points (on-policy baseline, no rejection).
  D  denial suppression: resample away honesty/denial sentences.

Usage:
  .venv/bin/python3 analysis/exp2_resample.py --smoke                 # template sanity check
  .venv/bin/python3 analysis/exp2_resample.py --arm C --n 30
Env (.env in project root): NEBIUS_API_KEY / FIREWORKS_API_KEY / OPENROUTER_API_KEY
"""
import argparse, json, os, re, sys, time, threading, queue
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from exp1_backtracking import RUNS, split_sentences, align_trajectory, STRONG

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "exp2_out")
os.makedirs(OUT, exist_ok=True)

DENIAL = re.compile(
    r"\b(honest|unbiased|objective|regardless of (the )?(bet|donation|threshold)"
    r"|not (let|allow) (the )?(bet|donation|threshold)|accuracy is paramount"
    r"|most accurate estimate|not (going to|gonna) (game|manipulate)|integrity)\b",
    re.IGNORECASE)

# --------------------------------------------------------------- provider
def load_env():
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v and k not in os.environ:
                    os.environ[k] = v

PROVIDERS = {
    "nebius":    ("https://api.studio.nebius.com/v1/completions", "NEBIUS_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1/completions", "FIREWORKS_API_KEY"),
    "openrouter":("https://openrouter.ai/api/v1/completions", "OPENROUTER_API_KEY"),
}

PIN = {}  # set at startup: {"order": ["deepinfra/fp4"], "allow_fallbacks": False}

def complete(provider, model, prompt, max_tokens=12000, temperature=1.0, top_p=1.0, tries=5):
    url, keyvar = PROVIDERS[provider]
    key = os.environ.get(keyvar)
    if not key:
        raise SystemExit(f"missing {keyvar} in .env")
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p}
    if provider == "openrouter" and PIN:
        body["provider"] = PIN
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            return d["choices"][0]["text"]
        except Exception as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)

TEMPLATES = {
    "chatml": lambda u, p: (f"<|im_start|>user\n{u}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n{p}"),
    "glm":    lambda u, p: f"[gMASK]<sop><|user|>\n{u}<|assistant|>\n<think>{p}",
    "deepseek": lambda u, p: f"<|User|>{u}<|Assistant|><think>\n{p}",
}
TEMPLATE = ["glm"]

def build_prompt(user_prompt, think_prefix):
    return TEMPLATES[TEMPLATE[0]](user_prompt, think_prefix)

# --------------------------------------------------------------- fork finding
def find_forks(reasoning, traj, threshold, condition):
    """Returns dict arm -> (prefix_text, fork_sentence). Fork = first sentence of the given
    class; prefix = everything before it."""
    sentences = split_sentences(reasoning)
    hits, _ = align_trajectory(sentences, traj, threshold)
    if len(hits) < 2:
        return {}
    est_by_sent = {}
    for si, off, val, _tj in hits:
        est_by_sent[si] = val
    bad_is_above = (condition == "below_good")
    side = None
    forks = {}
    for si, sent in enumerate(sentences):
        if si in est_by_sent:
            side = "above" if est_by_sent[si] > threshold else "below"
        is_bt = bool(STRONG.search(sent))
        is_denial = bool(DENIAL.search(sent))
        on_bad = side == ("above" if bad_is_above else "below")
        if is_bt and side is not None:
            arm = "A" if on_bad else "B"
            if arm not in forks and si > 0:
                forks[arm] = si
        if is_denial and "D" not in forks and si > 0:
            forks["D"] = si
        if len(forks) == 3:
            break
    out = {}
    for arm, si in forks.items():
        out[arm] = ("\n".join(sentences[:si]), sentences[si])
    return out

FINAL_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(million|billion|m\b|b\b)?", re.IGNORECASE)
MULT = {"million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "": 1.0, None: 1.0}

def parse_final(cont):
    """Final answer from a continuation: last plausible number in post-</think> text,
    else in the last 400 chars of the whole continuation."""
    tail = cont.split("</think>")[-1] if "</think>" in cont else cont[-400:]
    vals = [float(m.group(1).replace(",", "")) * MULT.get((m.group(2) or "").lower().strip(), 1.0)
            for m in FINAL_NUM.finditer(tail)]
    vals = [v for v in vals if 1e5 <= v <= 1e11]
    return vals[-1] if vals else None

# --------------------------------------------------------------- run
def run_arm(cfg, cond, arm, n_rollouts, n_traces, k_reject, smoke=False, plain=False):
    rd = os.path.join(RUNS, cfg["run"])
    thr = json.load(open(os.path.join(rd, "threshold.json")))["threshold"]
    trajs = json.load(open(os.path.join(rd, "trajectories.json")))
    data = json.load(open(os.path.join(rd, f"{cond}.json")))
    user_prompt, rows = data["prompt"], data["rows"]
    suffix = f"arm{arm}" + ("plain" if plain else "")
    outfile = os.path.join(OUT, f"{cfg['name']}_{cond}_{suffix}.jsonl")
    done = set()
    if os.path.exists(outfile):
        for line in open(outfile):
            j = json.loads(line)
            done.add((j["trace"], j["sample"]))
    lock = threading.Lock()
    jobs = queue.Queue()
    n_queued = 0
    for i, (row, traj) in enumerate(zip(rows, trajs[cond])):
        if n_queued >= n_traces:
            break
        r = row.get("reasoning") or ""
        if not r or not traj:
            continue
        forks = find_forks(r, traj, thr, cond)
        src = forks.get(arm)
        if not src:
            continue
        prefix, fork_sent = src
        n_queued += 1
        for s in range(n_rollouts):
            if (i, s) not in done:
                jobs.put((i, s, prefix, fork_sent))
    print(f"[{cond} {suffix}] traces={n_queued} jobs={jobs.qsize()} (resuming past {len(done)})")
    if smoke:
        i, s, prefix, fork_sent = jobs.get()
        print("---- fork sentence:", fork_sent[:200])
        print("---- prefix tail:", prefix[-400:])
        cont = complete(cfg["provider"], cfg["model"], build_prompt(user_prompt, prefix),
                        max_tokens=2000)
        print("---- continuation head:\n", cont[:1200])
        print("---- parsed final:", parse_final(cont))
        return

    def worker():
        while True:
            try:
                i, s, prefix, fork_sent = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                rejected = 0
                head = ""
                if not plain:
                    # two-stage rejection: probe short heads until first sentence is clean
                    for attempt in range(k_reject):
                        h = complete(cfg["provider"], cfg["model"],
                                     build_prompt(user_prompt, prefix), max_tokens=100)
                        first = split_sentences(h)[:1]
                        bad = bool(first) and (
                            (arm in ("A", "B") and STRONG.search(first[0])) or
                            (arm == "D" and DENIAL.search(first[0])))
                        if not bad:
                            head = h
                            break
                        rejected += 1
                    else:
                        head = h  # resilience cap hit; keep last
                cont = head + complete(cfg["provider"], cfg["model"],
                                       build_prompt(user_prompt, prefix + head))
                rec = {"trace": i, "sample": s, "rejected": rejected,
                       "final": parse_final(cont), "above": (parse_final(cont) or 0) > thr,
                       "cont_head": cont[:300], "cont_tail": cont[-300:]}
                with lock:
                    with open(outfile, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print("job failed", i, s, repr(e))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(cfg["conc"])]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print("done ->", outfile)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-122b-a10b_20260815_030702")
    ap.add_argument("--provider", default="nebius")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--arm", default="A", choices=list("ABD"))
    ap.add_argument("--plain", action="store_true", help="no rejection: on-policy baseline at same forks")
    ap.add_argument("--template", default="glm", choices=["glm", "chatml", "deepseek"])
    ap.add_argument("--cond", default="below_good")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--traces", type=int, default=15)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--conc", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pin", default=None, help="openrouter provider pin, e.g. deepinfra/fp4")
    a = ap.parse_args()
    load_env()
    TEMPLATE[0] = a.template
    if a.pin:
        PIN.update({"order": [a.pin], "allow_fallbacks": False})
    cfg = {"run": a.model, "name": a.model.split("_")[0], "provider": a.provider,
           "model": a.model_id, "conc": a.conc}
    run_arm(cfg, a.cond, a.arm, a.n, a.traces, a.k, smoke=a.smoke, plain=a.plain)
