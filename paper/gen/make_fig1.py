# -*- coding: utf-8 -*-
"""Figure 1: one benchmark instance (well A: GR + CARB; well B: GR) with the
ground-truth hiatus depths, plus the depth-domain spectrum of well A."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_v2")
IID = sys.argv[1] if len(sys.argv) > 1 else "test_000"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "fig1_instance.png")


def detrend(x, win=101):
    return x - np.convolve(x, np.ones(win) / win, mode="same")


def main():
    d = os.path.join(BENCH, "data", IID)
    a = pd.read_csv(os.path.join(d, "well_A.csv"))
    b = pd.read_csv(os.path.join(d, "well_B.csv"))
    truths = json.load(open(os.path.join(BENCH, "truths.json"), encoding="utf-8"))
    t = truths[IID]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})

    ax = axes[0]
    ax.plot(a.GR_API, a.Depth_m, lw=0.7, color="#2a78d6", label="GR")
    ax.plot(a.CARB_pct * 10, a.Depth_m, lw=0.7, color="#eb6834", alpha=0.8, label="CARB x10")
    for depth in t["Q3_hiatuses_depth_m"]:
        ax.axhline(depth, color="#898781", ls="--", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("counts")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"Well A  (hiatuses: {len(t['Q3_hiatuses_depth_m'])})")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(b.GR_API, b.Depth_m, lw=0.7, color="#2a78d6")
    ax.axhline(t["Q8"]["wellB_depth_m"], color="#1a9850", ls="--", lw=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("GR")
    ax.set_title("Well B  (regional hiatus dashed)")

    ax = axes[2]
    for col, color, label in (("GR_API", "#2a78d6", "GR"), ("CARB_pct", "#eb6834", "CARB")):
        x = detrend(a[col].values)
        f, p = periodogram(x, fs=2.0)
        wl = 1.0 / np.maximum(f, 1e-12)
        m = (wl >= 1) & (wl <= 80)
        ax.loglog(wl[m], p[m], lw=0.8, color=color, label=label)
    ax.axvline(405 * t["Q4_mean_SR_cm_kyr"] / 100, color="#898781", ls=":", lw=0.9)
    ax.set_xlabel("wavelength (m)")
    ax.set_ylabel("power")
    ax.set_title("Depth-domain spectrum (405-kyr dotted)")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
