# -*- coding: utf-8 -*-
"""Figure 2: macro accuracy by model and condition, with the reference solver line."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "exp", "results")
OUT = os.path.join(BASE, "figures", "fig2_results.png")

RUNS = {
    "Qwen3.5-122B": {"zero": "qwen35-122b_zero", "CoT": "qwen35-122b_cot", "Tool": "qwen35-122b_tool"},
    "DeepSeek-V3.2": {"zero": "deepseek-v3-2_zero", "CoT": "deepseek-v3-2_cot", "Tool": "deepseek-v3-2_tool"},
    "Qwen3-32B": {"zero": "qwen3-32b_zero", "CoT": "qwen3-32b_cot", "Tool": "qwen3-32b_tool"},
    "GLM-4.5-Air": {"zero": "glm45air_zero", "CoT": "glm45air_cot", "Tool": "glm45air_tool"},
    "Qwen3-14B": {"zero": "qwen3-14b_zero", "CoT": "qwen3-14b_cot"},
}
SOLVER = 0.275


def macro(tag):
    p = os.path.join(RESULTS, tag + ".jsonl")
    if not os.path.exists(p):
        return 0.0
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return sum((r.get("score") or {}).get("total", 0) for r in rows) / max(len(rows), 1) * 100


def main():
    models = list(RUNS)
    conds = ["zero", "CoT", "Tool"]
    colors = {"zero": "#2a78d6", "CoT": "#eb6834", "Tool": "#1a9850"}
    width = 0.26
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, cond in enumerate(conds):
        vals = [macro(RUNS[m].get(cond, "")) if cond in RUNS[m] else 0.0 for m in models]
        ax.bar(x + (k - 1) * width, vals, width, label=cond, color=colors[cond])
    ax.axhline(SOLVER, color="#898781", ls="--", lw=1.0, label="reference solver")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("macro accuracy (%)")
    ax.set_ylim(0, 32)
    ax.grid(axis="y", color="#e1e0d9", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
