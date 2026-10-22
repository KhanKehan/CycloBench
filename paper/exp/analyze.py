# -*- coding: utf-8 -*-
"""Analyse a CycloBench-v2 result file (JSONL from run_api/run_local/solver).

Reports per-question accuracy, common failure modes (the 41-kyr shortcut,
hiatus count mismatch, partial Q6), and error magnitudes for numeric questions.

Usage: python analyze.py results/qwen3_zero.jsonl [--bench ../bench_v2]
"""

import argparse
import json
import os

import numpy as np


def load_results(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--bench", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_v2"))
    args = ap.parse_args()

    truths = json.load(open(os.path.join(args.bench, "truths.json"), encoding="utf-8"))
    rows = load_results(args.results)

    per_q = {f"Q{i}": [] for i in range(1, 9)}
    shortcut_41 = 0
    count_mismatch = 0
    q6_partial = 0
    numeric_err = {"Q1": [], "Q2": [], "Q4": [], "Q5": []}

    for r in rows:
        ans = r.get("answer") or {}
        truth = truths[r["iid"]]
        sc = r.get("score") or {}
        for i in range(1, 9):
            per_q[f"Q{i}"].append(int(sc.get(f"Q{i}", 0)))

        q5 = ans.get("Q5_obliquity_kyr")
        if isinstance(q5, (int, float)) and abs(q5 - 41.0) < 1.0:
            shortcut_41 += 1
        q3 = ans.get("Q3_hiatuses")
        if isinstance(q3, list) and len(q3) != len(truth["Q3_hiatuses_depth_m"]):
            count_mismatch += 1
        q6 = ans.get("Q6")
        if isinstance(q6, dict):
            keys = ["p60_autogenic", "p112_autogenic", "carb_has_60kyr_peak", "carb_has_112kyr_peak"]
            got = sum(int(q6.get(k) == truth["Q6"][k]) for k in keys)
            if 0 < got < 4:
                q6_partial += 1
        for k, tk in (("Q1", "Q1_total_duration_Myr"), ("Q2", "Q2_age_250m_Ma"),
                      ("Q4", "Q4_mean_SR_cm_kyr"), ("Q5", "Q5_obliquity_kyr")):
            v = ans.get(tk)
            if isinstance(v, (int, float)):
                numeric_err[k].append(abs(v - truth[tk]))

    n = len(rows)
    print(f"results: {args.results}  n={n}")
    print()
    print("| question | pass rate |")
    print("|---|---|")
    for k, v in per_q.items():
        print(f"| {k} | {np.mean(v):.3f} |")
    print()
    print(f"macro: {np.mean([np.mean(v) for v in per_q.values()]):.4f}")
    print(f"41-kyr shortcut (Q5): {shortcut_41}/{n}")
    print(f"hiatus count mismatch (Q3): {count_mismatch}/{n}")
    print(f"partial Q6 (1-3 of 4 booleans): {q6_partial}/{n}")
    for k, errs in numeric_err.items():
        if errs:
            print(f"{k} median abs error: {np.median(errs):.3f}  p90: {np.percentile(errs, 90):.3f}")

    md = os.path.splitext(args.results)[0] + "_analysis.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Analysis: {os.path.basename(args.results)}\n\n")
        f.write(f"n = {n}\n\n| question | pass rate |\n|---|---|\n")
        for k, v in per_q.items():
            f.write(f"| {k} | {np.mean(v):.3f} |\n")
        f.write(f"\nmacro = {np.mean([np.mean(v) for v in per_q.values()]):.4f}\n")
    print("wrote", md)


if __name__ == "__main__":
    main()
