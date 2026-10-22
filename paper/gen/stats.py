# -*- coding: utf-8 -*-
"""Summary statistics of a generated benchmark (for the paper's data table)."""

import json
import os
import sys

import numpy as np
import pandas as pd


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_v2")
    truths = json.load(open(os.path.join(bench, "truths.json"), encoding="utf-8"))
    rows = []
    for iid, t in truths.items():
        rows.append({
            "split": t["meta"]["split"],
            "T_Myr": t["Q1_total_duration_Myr"],
            "depth_A": t["meta"]["depth_A_m"],
            "depth_B": t["meta"]["depth_B_m"],
            "SR": t["Q4_mean_SR_cm_kyr"],
            "n_hiatus_A": len(t["Q3_hiatuses_depth_m"]),
            "n_hiatus_B": t["meta"]["n_hiatus_B"],
            "hiatus_dur_mean": float(np.mean(t["Q3_hiatuses_duration_kyr"])),
            "obliquity": t["Q5_obliquity_kyr"],
        })
    df = pd.DataFrame(rows)
    print(df.groupby("split")[["T_Myr", "depth_A", "depth_B", "SR", "n_hiatus_A", "n_hiatus_B",
                               "hiatus_dur_mean", "obliquity"]].describe().round(2).to_string())
    out = {
        "n_instances": int(len(df)),
        "n_items": int(len(df) * 8),
        "split": df.groupby("split").size().to_dict(),
        "T_Myr": [float(df.T_Myr.min()), float(df.T_Myr.max())],
        "depth_A_m": [float(df.depth_A.min()), float(df.depth_A.max())],
        "SR_cm_kyr": [float(df.SR.min()), float(df.SR.max())],
        "hiatus_A": [int(df.n_hiatus_A.min()), int(df.n_hiatus_A.max())],
        "hiatus_dur_kyr": [float(df.hiatus_dur_mean.min()), float(df.hiatus_dur_mean.max())],
        "obliquity_kyr": [float(df.obliquity.min()), float(df.obliquity.max())],
    }
    with open(os.path.join(bench, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
