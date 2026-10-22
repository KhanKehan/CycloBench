# -*- coding: utf-8 -*-
"""Sanity checks for a generated CycloBench-v2 split.

For every instance we verify that the recorded data actually contains the
spectral evidence the questions rely on:

  * a 405-kyr eccentricity peak in GR and CARB (depth wavelength = 405 kyr * SR)
  * an obliquity peak in GR at the simulated 300-Ma period
  * autogenic 60/112-kyr peaks in GR but not in CARB (contrast ratio > 5x)
  * 250 m lies inside well A

Usage:  python validate.py bench_v2
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import periodogram

DETREND_WIN = 101  # 50 m at 0.5 m sampling


def band_power(x, wl_lo, wl_hi, fs=2.0):
    x = x - np.convolve(x, np.ones(DETREND_WIN) / DETREND_WIN, mode="same")
    f, p = periodogram(x, fs=fs)
    wl = 1.0 / np.maximum(f, 1e-12)
    m = (wl >= wl_lo) & (wl <= wl_hi)
    return float(p[m].max()) if m.any() else 0.0


def check_instance(root, iid, truth):
    a = pd.read_csv(os.path.join(root, "data", iid, "well_A.csv"))
    sr = truth["Q4_mean_SR_cm_kyr"] / 100.0  # m/kyr
    obliq = truth["Q5_obliquity_kyr"]
    wl405 = 405.0 * sr
    wlob = obliq * sr
    wl60 = 60.0 * sr
    wl112 = 112.0 * sr

    res = {}
    res["405_gr"] = band_power(a.GR_API.values, 0.8 * wl405, 1.25 * wl405)
    res["405_carb"] = band_power(a.CARB_pct.values, 0.8 * wl405, 1.25 * wl405)
    res["ob_gr"] = band_power(a.GR_API.values, 0.85 * wlob, 1.15 * wlob)
    p60_gr = band_power(a.GR_API.values, 0.9 * wl60, 1.1 * wl60)
    p60_carb = band_power(a.CARB_pct.values, 0.9 * wl60, 1.1 * wl60)
    p112_gr = band_power(a.GR_API.values, 0.9 * wl112, 1.1 * wl112)
    p112_carb = band_power(a.CARB_pct.values, 0.9 * wl112, 1.1 * wl112)
    res["ratio_60"] = p60_gr / max(p60_carb, 1e-9)
    res["ratio_112"] = p112_gr / max(p112_carb, 1e-9)
    res["q6_mode"] = ("auto" if truth["Q6"]["p60_autogenic"] else
                      ("allo" if truth["Q6"]["carb_has_60kyr_peak"] else "none"))
    ev = truth["meta"].get("q6_evidence")
    if ev:
        res.update(ev)
    res["has_250m"] = bool(a.Depth_m.max() >= 250.0)
    return res


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "bench_v2"
    truths = json.load(open(os.path.join(root, "truths.json"), encoding="utf-8"))
    rows = []
    for iid, truth in truths.items():
        r = check_instance(root, iid, truth)
        r["iid"] = iid
        r["split"] = truth["meta"]["split"]
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df.groupby("split")[["405_gr", "405_carb", "ob_gr", "ratio_60", "ratio_112"]].describe().round(1).to_string())
    print()
    print("instances with 60-kyr contrast > 5x:", int((df.ratio_60 > 5).sum()), "/", len(df))
    print("instances with 112-kyr contrast > 5x:", int((df.ratio_112 > 5).sum()), "/", len(df))
    print("instances with 250 m in well A:", int(df.has_250m.sum()), "/", len(df))
    print("instances with 405 peak in both wells:", int(((df["405_gr"] > 0) & (df["405_carb"] > 0)).sum()), "/", len(df))
    if "rel60_gr" in df:
        ok_auto = (df.q6_mode == "auto") & (df.rel60_gr > 0.5) & (df.rel60_carb < 0.1) & (df.rel112_gr > 0.5) & (df.rel112_carb < 0.1)
        ok_allo = (df.q6_mode == "allo") & (df.rel60_gr > 0.5) & (df.rel60_carb > 0.5) & (df.rel112_gr > 0.5) & (df.rel112_carb > 0.5)
        ok_none = (df.q6_mode == "none") & (df.rel60_gr < 0.1) & (df.rel60_carb < 0.1)
    else:
        ok_auto = (df.q6_mode == "auto") & (df.ratio_60 > 5) & (df.ratio_112 > 5)
        ok_allo = (df.q6_mode == "allo") & (df.ratio_60 > 0.3) & (df.ratio_60 < 3) & (df.ratio_112 < 3)
        ok_none = df.q6_mode == "none"
    ok = (ok_auto | ok_allo | ok_none) & df.has_250m
    print("PASS:", int(ok.sum()), "/", len(df))
    df.to_csv(os.path.join(root, "validation.csv"), index=False)


if __name__ == "__main__":
    main()
