# -*- coding: utf-8 -*-
"""Run a model through an OpenAI-compatible chat API on CycloBench-v2.

Examples
--------
DASHSCOPE_API_KEY=... python run_api.py --model qwen3-32b --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --out results/qwen3-32b_zero.jsonl
DEEPSEEK_API_KEY=...   python run_api.py --model deepseek-chat --base-url https://api.deepseek.com/v1 --out results/deepseek_zero.jsonl

Conditions: zero_shot | cot | tool (tool runs up to N code turns).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from checker import score
from prompts import build_prompt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_json(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


NO_THINKING = False


def chat(base_url, api_key, model, messages, temperature=0.0, max_tokens=1200):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if NO_THINKING:
        payload["enable_thinking"] = False
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=240)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(last)


DENY = ("import os", "import sys", "subprocess", "socket", "shutil", "pathlib",
        "__import__", "open(", "eval(", "exec(", "compile(", "globals(", "locals(",
        "ctypes", "pickle", "requests", "urllib", "os.", "sys.")

CHILD = r'''
import sys, io, contextlib, json
import pandas as pd, numpy as np
import scipy.signal as signal, scipy.stats as stats
CODE = {code!r}
DATA = {data!r}
A = pd.read_csv(DATA + "/well_A.csv")
B = pd.read_csv(DATA + "/well_B.csv")
_ALLOWED = ("numpy", "pandas", "scipy", "math", "statistics", "itertools", "collections")
_real_import = __import__
def _safe_import(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED:
        raise ImportError("module not allowed: " + name)
    return _real_import(name, *a, **k)
_safe = {{k: getattr(__builtins__, k) for k in ("len","range","min","max","abs","round","sum","sorted","enumerate","zip","list","dict","set","tuple","float","int","str","bool","print") if hasattr(__builtins__, k)}}
_safe["__import__"] = _safe_import
_env = {{"__builtins__": _safe, "pd": pd, "np": np, "signal": signal, "stats": stats, "A": A, "B": B}}
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        exec(compile(CODE, "<model>", "exec"), _env)
    sys.stdout.write(_buf.getvalue()[-4000:] or "[no output]")
except Exception as e:
    sys.stdout.write("ERROR " + type(e).__name__ + ": " + str(e) + "\n" + _buf.getvalue()[-2000:])
'''


def run_code(code, data_dir, timeout=120):
    low = code.lower()
    for bad in DENY:
        if bad in low:
            return f"[blocked: contains {bad!r}]"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(CHILD.format(code=code, data=data_dir.replace("\\", "/")))
        path = f.name
    try:
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
               "PYTHONIOENCODING": "utf-8"}
        out = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout, env=env)
        text = (out.stdout or "")[-4000:]
        if out.stderr:
            text += "\n[stderr]\n" + out.stderr[-800:]
        return text
    except subprocess.TimeoutExpired:
        return "[timeout]"
    finally:
        os.unlink(path)


def solve_one(base_url, api_key, model, instance_dir, condition, tool_turns=3):
    a = pd.read_csv(os.path.join(instance_dir, "well_A.csv"))
    b = pd.read_csv(os.path.join(instance_dir, "well_B.csv"))
    prompt = build_prompt(instance_dir, a, b, condition)
    messages = [{"role": "user", "content": prompt}]
    answer = None
    raw = []
    for turn in range(tool_turns if condition == "tool" else 1):
        text = chat(base_url, api_key, model, messages, max_tokens=2048)
        raw.append(text)
        if condition == "tool":
            blocks = re.findall(r"```python\s*(.*?)```", text, re.S)
            if blocks:
                result = run_code(blocks[-1], instance_dir)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Execution result:\n{result}\n\nNow return the final JSON."})
                continue
        answer = extract_json(text)
        if answer is not None:
            break
    if answer is None and condition == "tool":
        messages.append({"role": "user", "content": "No more code. Return the final JSON object now."})
        text = chat(base_url, api_key, model, messages, max_tokens=2048)
        raw.append(text)
        answer = extract_json(text)
    return answer, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--condition", default="zero_shot", choices=["zero_shot", "cot", "tool"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global NO_THINKING
    NO_THINKING = bool(args.no_thinking)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing env {args.api_key_env}")

    bench = os.path.join(ROOT, "bench_v2")
    truths = json.load(open(os.path.join(bench, "truths.json"), encoding="utf-8"))
    ids = [k for k, v in truths.items() if v["meta"]["split"] == args.split]
    ids.sort()
    if args.limit:
        ids = ids[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []

    def task(iid):
        try:
            ans, raw = solve_one(args.base_url, api_key, args.model, os.path.join(bench, "data", iid), args.condition)
            sc = score(ans, truths[iid]) if ans else {"total": 0.0}
            return {"iid": iid, "answer": ans, "score": sc, "raw": raw}
        except Exception as e:  # noqa: BLE001
            return {"iid": iid, "answer": None, "score": {"total": 0.0}, "error": repr(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, i): i for i in ids}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            results.append(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{i + 1}/{len(ids)}] {r['iid']} score={r['score'].get('total', 0):.3f}", flush=True)
    out_f.close()
    macro = sum(r["score"].get("total", 0) for r in results) / max(len(results), 1)
    print(json.dumps({"model": args.model, "condition": args.condition, "n": len(results), "macro": round(macro, 4)}, indent=2))


if __name__ == "__main__":
    main()
