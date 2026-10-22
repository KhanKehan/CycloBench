# -*- coding: utf-8 -*-
"""Local HF evaluation on CycloBench-v2 (used for base vs GRPO adapters).

python run_local.py --model Qwen/Qwen2.5-1.5B-Instruct --split test --out results/qwen25-1.5b_base.jsonl
python run_local.py --model outputs/grpo-cyclobench --split test --out results/grpo_test.jsonl
"""

import argparse
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from checker import score
from prompts import build_prompt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_json(text):
    import re

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


@torch.inference_mode()
def generate(model, tok, prompt, max_new_tokens=1024):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    bench = os.path.join(ROOT, "bench_v2")
    truths = json.load(open(os.path.join(bench, "truths.json"), encoding="utf-8"))
    ids = sorted(k for k, v in truths.items() if v["meta"]["split"] == args.split)
    if args.limit:
        ids = ids[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for i, iid in enumerate(ids):
        d = os.path.join(bench, "data", iid)
        a = pd.read_csv(os.path.join(d, "well_A.csv"))
        b = pd.read_csv(os.path.join(d, "well_B.csv"))
        prompt = build_prompt(d, a, b, "zero_shot")
        text = generate(model, tok, prompt, args.max_new_tokens)
        ans = extract_json(text)
        sc = score(ans, truths[iid]) if ans else {"total": 0.0}
        rows.append({"iid": iid, "answer": ans, "score": sc, "raw": text})
        print(f"[{i + 1}/{len(ids)}] {iid} {sc.get('total', 0):.3f}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    macro = sum(r["score"].get("total", 0) for r in rows) / max(len(rows), 1)
    print(json.dumps({"model": args.model, "n": len(rows), "macro": round(macro, 4)}, indent=2))


if __name__ == "__main__":
    main()
