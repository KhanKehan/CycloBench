# -*- coding: utf-8 -*-
"""Bootstrap CIs and per-question pass rates for the v2.2 matrix."""

import glob
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
BENCH = os.path.join(BASE, "..", "bench_v2")

RUNS = {
    "Qwen3.5-122B": {"zero": "qwen35-122b_zero", "CoT": "qwen35-122b_cot", "tool": "qwen35-122b_tool"},
    "DeepSeek-V3.2": {"zero": "deepseek-v3-2_zero", "CoT": "deepseek-v3-2_cot", "tool": "deepseek-v3-2_tool"},
    "Qwen3-14B": {"zero": "qwen3-14b_zero", "CoT": "qwen3-14b_cot"},
    "GLM-4.5-Air": {"zero": "glm45air_zero", "CoT": "glm45air_cot"},
    "Qwen3-32B": {"zero": "qwen3-32b_zero", "CoT": "qwen3-32b_cot", "tool": "qwen3-32b_tool"},
}


def load(tag):
    p = os.path.join(RESULTS, tag + ".jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows


def per_instance(rows):
    return np.array([(r.get("score") or {}).get("total", 0.0) for r in rows])


def boot_ci(x, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = x[idx].mean()
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print("=== main table rows with 95% bootstrap CI (percent) ===")
    for model, conds in RUNS.items():
        cells = []
        for cond in ("zero", "CoT", "tool"):
            if cond not in conds:
                cells.append("--")
                continue
            x = per_instance(load(conds[cond]))
            m, lo, hi = boot_ci(x)
            cells.append(f"{m*100:.2f} [{lo*100:.1f}, {hi*100:.1f}]")
        print(f"{model:14s} & " + " & ".join(cells) + r" \\")

    print()
    print("=== per-question pass rate, zero-shot (percent) ===")
    for model, conds in RUNS.items():
        rows = load(conds["zero"])
        per = []
        for q in range(1, 9):
            vals = [(r.get("score") or {}).get(f"Q{q}", 0) for r in rows]
            per.append(f"{np.mean(vals)*100:.1f}")
        print(f"{model:14s} & " + " & ".join(per) + r" \\")


if __name__ == "__main__":
    main()
