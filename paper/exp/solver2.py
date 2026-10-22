# -*- coding: utf-8 -*-
"""Reference solver v2: sliding-window phase tracking (astrochron-style).

Instead of one FFT peak, we fit sin/cos at the local 405-kyr wavelength in
overlapping depth windows, unwrap the phase, and read:

    local wavelength  wl(d) = 2*pi / (dphi/dd)
    local rate        SR(d) = wl(d) / 405          [m/kyr]
    total time        T     = 405 * dphi_total/2pi [kyr]
    hiatus           jumps in dphi between windows

then answer the eight questions and score against the stored truth.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import periodogram

FS = 2.0  # 0.5 m sampling


def detrend(x, win=101):
    return x - np.convolve(x, np.ones(win) / win, mode="same")


def spectrum(x):
    f, p = periodogram(detrend(x), fs=FS)
    return 1.0 / np.maximum(f, 1e-12), p


def dominant_405(x):
    wl, p = spectrum(x)
    m = (wl >= 15) & (wl <= 60)
    return float(wl[m][np.argmax(p[m])]) if m.any() else 28.0


def phase_track(depth, x, wl, win_m=80.0, step_m=4.0):
    """Fit sin/cos at wavelength `wl` in overlapping windows; return (d, phase, amp).

    The fit uses window-local coordinates so the phase advances with depth and can
    be unwrapped without aliasing (step < wl/2).
    """
    win = int(win_m / 0.5)
    step = int(step_m / 0.5)
    k = 2 * np.pi / wl
    centers, phases, amps = [], [], []
    for i in range(0, len(depth) - win, step):
        d = depth[i:i + win]
        y = x[i:i + win]
        u = d - d.mean()
        s, c = np.sin(k * u), np.cos(k * u)
        X = np.column_stack([s, c, np.ones_like(u)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coef[0], coef[1]
        phases.append(np.arctan2(b, a))
        amps.append(np.hypot(a, b))
        centers.append(float(np.mean(d)))
    return np.array(centers), np.unwrap(np.array(phases)), np.array(amps)


def find_jumps(centers, phase, wl, amp):
    """A hiatus produces a phase step larger than the expected window advance."""
    dphi = np.diff(phase)
    expected = 2 * np.pi * np.diff(centers) / wl
    resid = dphi - expected
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-6
    jumps = []
    i = 0
    while i < len(resid):
        if abs(resid[i] - med) > max(8 * mad, 1.0):
            j = i
            while j + 1 < len(resid) and abs(resid[j + 1] - med) > max(8 * mad, 1.0):
                j += 1
            step = float((resid[i:j + 1] - med).sum())
            dur = abs(step) / (2 * np.pi) * 405.0
            if 40 <= dur <= 800:
                jumps.append({"depth_m": round(float(centers[i]), 1), "duration_kyr": round(dur, 0)})
            i = j + 1
        else:
            i += 1
    return jumps


def solve(a, b):
    dA, grA, carbA = a.Depth_m.values, a.GR_API.values, a.CARB_pct.values
    dB, grB = b.Depth_m.values, b.GR_API.values

    wl = dominant_405(grA)
    centers, phase, amp = phase_track(dA, grA, wl)
    if len(centers) < 3:
        return None

    # total time from the full phase advance (includes hiatus time)
    T_kyr = 405.0 * (phase[-1] - phase[0]) / (2 * np.pi)
    T = abs(T_kyr) / 1000.0

    jumps = find_jumps(centers, phase, wl, amp)

    # local SR from the phase slope between windows
    dphi = np.diff(phase)
    dd = np.diff(centers)
    ok = np.abs(dphi) > 1e-6
    sr_local = np.abs(dd[ok] / (dphi[ok] / (2 * np.pi) * 405.0))  # m/kyr
    sr = float(np.median(sr_local)) if len(sr_local) else wl / 405.0

    mean_sr = float(dA.max()) / (T * 1000.0) * 100.0 if T > 0 else 0.0

    # time-depth model from cumulative phase
    t_cum = 405.0 * (phase - phase[0]) / (2 * np.pi)  # kyr, can be negative by sign
    t_cum = t_cum - t_cum[0]
    order = np.argsort(centers)
    t_at = np.interp(250.0, centers[order], t_cum[order]) if 250.0 <= dA.max() else 250.0 * 405.0 / wl
    age250 = 300.0 + abs(t_at) / 1000.0

    # obliquity
    wls, ps = spectrum(grA)
    m = (wls >= 0.8) & (wls <= 4.5)
    ob_wl = float(wls[m][np.argmax(ps[m])]) if m.any() else 2.2
    ob_kyr = ob_wl / sr

    # Q6 band contrast at 60/112 kyr
    wlc, pc = spectrum(carbA)

    def power(pw, w, lo=0.9, hi=1.1):
        mm = (pw[0] >= lo * w) & (pw[0] <= hi * w)
        return float(np.max(pw[1][mm])) if mm.any() else 0.0

    gr_spec = (wls, ps)
    p60_gr, p60_c = power(gr_spec, 60 * sr), power((wlc, pc), 60 * sr)
    p112_gr, p112_c = power(gr_spec, 112 * sr), power((wlc, pc), 112 * sr)
    q6 = {
        "p60_autogenic": p60_gr > 3 * p60_c,
        "p112_autogenic": p112_gr > 3 * p112_c,
        "carb_has_60kyr_peak": p60_c > 0.5 * p60_gr,
        "carb_has_112kyr_peak": p112_c > 0.5 * p112_gr,
    }

    # Q7 modulation from the obliquity band envelope (bandpass via FFT mask)
    def mod_index(x):
        f = np.fft.rfft(detrend(x))
        freqs = np.fft.rfftfreq(len(x), d=0.5)
        wl_ = 1.0 / np.maximum(freqs, 1e-12)
        mask = (wl_ >= ob_wl * 0.8) & (wl_ <= ob_wl * 1.25)
        f2 = np.where(mask, f, 0)
        y = np.fft.irfft(f2, n=len(x))
        env = np.abs(y)
        return float(np.std(env) / (np.mean(env) + 1e-9))

    mi_gr, mi_c = mod_index(grA), mod_index(carbA)

    # Q8 regional hiatus: match durations between wells
    centers_b, phase_b, amp_b = phase_track(dB, grB, dominant_405(grB))
    jumps_b = find_jumps(centers_b, phase_b, wl, amp_b)
    q8 = {"wellA_depth_m": 0.0, "wellB_depth_m": 0.0, "duration_kyr": 0.0}
    for h in jumps:
        for hb in jumps_b:
            if abs(hb["duration_kyr"] - h["duration_kyr"]) <= 60:
                q8 = {"wellA_depth_m": h["depth_m"], "wellB_depth_m": hb["depth_m"],
                      "duration_kyr": h["duration_kyr"]}
                break
        if q8["wellA_depth_m"]:
            break

    return {
        "Q1_total_duration_Myr": round(T, 3),
        "Q2_age_250m_Ma": round(age250, 3),
        "Q3_hiatuses": jumps[:6],
        "Q4_mean_SR_cm_kyr": round(mean_sr, 3),
        "Q5_obliquity_kyr": round(ob_kyr, 1),
        "Q6": q6,
        "Q7": {"exists": True, "stronger_in": "CARB" if mi_c > mi_gr else "GR",
               "GR_factor": round(min(max(mi_gr, 0.05), 1.5), 2),
               "CARB_factor": round(min(max(mi_c, 0.05), 1.5), 2)},
        "Q8": q8,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="../bench_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/solver2_test.jsonl")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.abspath(os.path.join(here, args.bench))
    truths = json.load(open(os.path.join(bench, "truths.json"), encoding="utf-8"))
    ids = sorted(k for k, v in truths.items() if v["meta"]["split"] == args.split)
    if args.limit:
        ids = ids[: args.limit]

    from checker import score

    rows = []
    for iid in ids:
        d = os.path.join(bench, "data", iid)
        a = pd.read_csv(os.path.join(d, "well_A.csv"))
        b = pd.read_csv(os.path.join(d, "well_B.csv"))
        try:
            ans = solve(a, b)
        except Exception as e:  # noqa: BLE001
            ans = None
            print(iid, "solver error", repr(e))
        sc = score(ans, truths[iid]) if ans else {"total": 0.0}
        rows.append({"iid": iid, "answer": ans, "score": sc})
        print(iid, {k: v for k, v in sc.items() if k != "total"}, f"total={sc['total']:.3f}")

    out = os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    macro = sum(r["score"]["total"] for r in rows) / max(len(rows), 1)
    print(json.dumps({"solver": "python-phase-v2", "split": args.split, "n": len(rows), "macro": round(macro, 4)}, indent=2))


if __name__ == "__main__":
    main()
