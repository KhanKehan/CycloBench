# -*- coding: utf-8 -*-
"""Re-score existing result files at scaled tolerances.

The 1x tolerances are tight (0.10 Myr on a 5-8 Myr record, 3 m on hiatus depth).
A reviewer can reasonably ask whether the near-zero model scores are an artefact of
that choice. This script answers it: same answers, tolerances x1, x2, x5, x10.

Usage: python tolerance_sweep.py
"""

import glob
import json
import os

import checker

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
BENCH = os.path.join(BASE, "..", "bench_v2")

RUNS = {
    "Qwen3.5-122B": {"zero": "qwen35-122b_zero", "cot": "qwen35-122b_cot", "tool": "qwen35-122b_tool"},
    "DeepSeek-V3.2": {"zero": "deepseek-v3-2_zero", "cot": "deepseek-v3-2_cot", "tool": "deepseek-v3-2_tool"},
    "Qwen3-32B": {"zero": "qwen3-32b_zero", "cot": "qwen3-32b_cot", "tool": "qwen3-32b_tool"},
    "GLM-4.5-Air": {"zero": "glm45air_zero", "cot": "glm45air_cot", "tool": "glm45air_tool"},
    "Qwen3-14B": {"zero": "qwen3-14b_zero", "cot": "qwen3-14b_cot"},
}


def rescore(rows, truths, factor):
    old = dict(checker.TOL)
    for k in checker.TOL:
        checker.TOL[k] = old[k] * factor
    try:
        out = []
        for r in rows:
            ans = r.get("answer")
            out.append(checker.score(ans, truths[r["iid"]])["total"] if ans else 0.0)
        return sum(out) / max(len(out), 1)
    finally:
        checker.TOL.update(old)


def main():
    truths = json.load(open(os.path.join(BENCH, "truths.json"), encoding="utf-8"))
    print(f"{'model':14s} {'cond':5s} " + " ".join(f"x{f:<5d}" for f in (1, 2, 5, 10)))
    for model, conds in RUNS.items():
        for cond, tag in conds.items():
            p = os.path.join(RESULTS, tag + ".jsonl")
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            cells = [rescore(rows, truths, f) for f in (1, 2, 5, 10)]
            print(f"{model:14s} {cond:5s} " + " ".join(f"{c*100:5.1f}" for c in cells))


if __name__ == "__main__":
    main()
