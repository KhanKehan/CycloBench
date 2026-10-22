# -*- coding: utf-8 -*-
"""CycloBench-v2 synthetic generator.

Forward model
-------------
A lacustrine basin at ~300 Ma is simulated on a 1-kyr time grid:

    depth(t) = integral of SR(t) dt          (SR in cm/kyr, depth in m)
    SR(t)    = smooth low-frequency profile + step, zero during hiatus events

Two wells share one regional hiatus but differ in sedimentation rate and in
their local hiatus events. Signals are mixtures of orbital components
(405/100-kyr eccentricity, obliquity at the 300-Ma period, precession) plus an
AR(1) background. GR carries autogenic 60/112-kyr cycles; CARB does not, and its
obliquity amplitude is more strongly modulated (~170-kyr) than GR.

Ground truth is computed from the latent parameters, not from the solver.

Output:
    data/<iid>/well_A.csv   Depth_m, GR_API, CARB_pct
    data/<iid>/well_B.csv   Depth_m, GR_API
    truths.json             <iid>: {Q1..Q8, meta}
    params.json             <iid>: latent parameters
"""

import argparse
import json
import math
import os
import zlib
from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import periodogram

DT = 1.0  # kyr


@dataclass
class InstanceParams:
    iid: str
    seed: int
    T_myr: float
    sr_base: float
    sr_amp1: float
    sr_amp2: float
    sr_step_at: float
    sr_step: float
    obliq_kyr: float
    prec_kyr: float
    hiatus_times: list
    hiatus_durs: list
    regional_index: int
    b_local_times: list
    b_local_durs: list
    gr_mod: float
    carb_mod: float
    aut60: float
    aut112: float
    noise_gr: float
    noise_carb: float
    trend: float
    q6_mode: str = "auto"
    q7_stronger: str = "CARB"


def _sin(t, period_kyr, phase=0.0):
    return np.sin(2 * np.pi * t / period_kyr + phase)


def _ar1(n, phi, sigma, rng):
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _sr_profile(t, p, offset):
    sr = p.sr_base + offset
    sr = sr + p.sr_amp1 * np.sin(2 * np.pi * t / 900.0 + p.seed % 7)
    sr = sr + p.sr_amp2 * np.sin(2 * np.pi * t / 310.0 + (2 * p.seed) % 7)
    sr = sr + np.where(t >= p.sr_step_at, p.sr_step, 0.0)
    return np.clip(sr, 2.5, 14.0)


def _mask(n, times, durs):
    m = np.ones(n, dtype=bool)
    for t0, dur in zip(times, durs):
        m &= ~((np.arange(n) >= t0) & (np.arange(n) < t0 + dur))
    return m


def _resample_to_depth(depth, t, signal):
    dmax = math.floor(depth.max() * 2) / 2
    dgrid = np.arange(0.0, dmax + 1e-9, 0.5)
    idx = np.clip(np.searchsorted(depth, dgrid, side="right"), 1, len(t) - 1)
    t_at_d = np.empty_like(dgrid)
    for k in range(len(dgrid)):
        j = idx[k]
        d0, d1 = depth[j - 1], depth[j]
        t_at_d[k] = t[j - 1] + (dgrid[k] - d0) / (d1 - d0) * (t[j] - t[j - 1]) if d1 > d0 else t[j]
    return dgrid, np.interp(t_at_d, t, signal), t_at_d


def _depth_at_time(depth, t, t0):
    j = min(int(np.searchsorted(t, t0)), len(depth) - 1)
    return float(depth[j])


def make_instance(p: InstanceParams):
    rng = np.random.default_rng(p.seed)
    n = int(round(p.T_myr * 1000.0 / DT))
    t = np.arange(n, dtype=float) * DT

    mask_a = _mask(n, p.hiatus_times, p.hiatus_durs)
    b_times = sorted([p.hiatus_times[p.regional_index]] + list(p.b_local_times))
    b_durs = [p.hiatus_durs[p.regional_index]] + list(p.b_local_durs)
    mask_b = _mask(n, b_times, b_durs)

    out = {}
    for well, offset, phase, mask in (("A", 0.0, 0.0, mask_a), ("B", 0.9, 1.7, mask_b)):
        sr = _sr_profile(t, p, offset)
        sr_eff = np.where(mask, sr, 0.0)
        depth = np.cumsum(sr_eff * DT / 100.0)

        f405 = _sin(t, 405.0, phase)
        f100 = _sin(t, 100.0, phase + 0.6)
        fob = _sin(t, p.obliq_kyr, phase + 1.1)
        fpre = _sin(t, p.prec_kyr, phase + 2.3)

        gr = (
            0.55 * f405
            + 0.30 * fob * (1.0 + p.gr_mod * _sin(t, 170.0, phase))
            + 0.12 * fpre
            + (0.0 if p.q6_mode == "none" else p.aut60) * _sin(t, 60.0, phase)
            + (0.0 if p.q6_mode == "none" else p.aut112) * _sin(t, 112.0, phase + 0.9)
            + p.trend * (t / t[-1])
            + _ar1(n, 0.55, p.noise_gr, rng)
        )
        carb = (
            0.45 * f405
            + 0.26 * fob * (1.0 + p.carb_mod * _sin(t, 170.0, phase + 0.4))
            + 0.10 * fpre
            + _ar1(n, 0.55, p.noise_carb, rng)
        )
        if p.q6_mode == "allo":
            carb = carb + p.aut60 * _sin(t, 60.0, phase) + p.aut112 * _sin(t, 112.0, phase + 0.9)

        dgrid, gr_d, _ = _resample_to_depth(depth, t, gr)
        _, carb_d, _ = _resample_to_depth(depth, t, carb)
        gr_d = 60.0 + 18.0 * (gr_d - gr_d.mean()) / (gr_d.std() + 1e-9)
        carb_d = 12.0 + 6.0 * (carb_d - carb_d.mean()) / (carb_d.std() + 1e-9)
        out[well] = dict(depth=dgrid, gr=gr_d, carb=carb_d, depth_t=depth, sr=sr_eff, gr_t=gr, carb_t=carb)

    a, b = out["A"], out["B"]
    hd_a = [round(_depth_at_time(a["depth_t"], t, t0), 2) for t0 in p.hiatus_times]
    hd_b = [round(_depth_at_time(b["depth_t"], t, t0), 2) for t0 in p.hiatus_times]

    total_depth = float(a["depth_t"][-1])
    # age at 250 m: interpolate the time-depth curve
    j = int(np.searchsorted(a["depth"], 250.0))
    j = max(1, min(j, len(a["depth"]) - 1))
    d0, d1 = a["depth"][j - 1], a["depth"][j]
    frac = (250.0 - d0) / (d1 - d0) if d1 > d0 else 0.0
    t250 = (j - 1 + frac) * 0.5 / max(a["sr"][j] / 100.0, 1e-9)
    # exact: use the time-depth mapping already computed
    _, _, t_at_depth = _resample_to_depth(a["depth_t"], t, np.zeros_like(t))
    t250 = float(np.interp(250.0, a["depth"], t_at_depth))

    ri = p.regional_index
    truth = {
        "Q1_total_duration_Myr": round(p.T_myr, 3),
        "Q2_age_250m_Ma": round(300.0 + t250 / 1000.0, 3),
        "Q3_hiatuses_depth_m": hd_a,
        "Q3_hiatuses_duration_kyr": [float(x) for x in p.hiatus_durs],
        "Q4_mean_SR_cm_kyr": round(total_depth * 100.0 / (p.T_myr * 1000.0), 3),
        "Q5_obliquity_kyr": round(p.obliq_kyr, 1),
        "Q6": {
            "p60_autogenic": bool(p.q6_mode == 'auto'),
            "p112_autogenic": bool(p.q6_mode == 'auto'),
            "carb_has_60kyr_peak": bool(p.q6_mode == 'allo'),
            "carb_has_112kyr_peak": bool(p.q6_mode == 'allo'),
        },
        "Q7": {
            "exists": True,
            "stronger_in": p.q7_stronger,
            "GR_factor": round(p.gr_mod, 2),
            "CARB_factor": round(p.carb_mod, 2),
        },
        "Q8": {
            "wellA_depth_m": hd_a[ri],
            "wellB_depth_m": hd_b[ri],
            "duration_kyr": float(p.hiatus_durs[ri]),
        },
        "meta": {
            "seed": p.seed,
            "depth_A_m": round(float(a["depth_t"][-1]), 1),
            "depth_B_m": round(float(b["depth_t"][-1]), 1),
            "n_hiatus_A": len(p.hiatus_times),
            "n_hiatus_B": len(b_times),
            "sr_base": round(p.sr_base, 2),
            "noise_gr": round(p.noise_gr, 3),
            "q6_evidence": _q6_evidence(out["A"], p),
        },
    }
    return out, truth


def _band_power_time(x, period_kyr, fs=1.0):
    x = x - np.convolve(x, np.ones(201) / 201, mode="same")
    f, pw = periodogram(x, fs=fs)
    wl = 1.0 / np.maximum(f, 1e-12)
    m = (wl >= period_kyr * 0.8) & (wl <= period_kyr * 1.25)
    return float(pw[m].max()) if m.any() else 1e-9


def _q6_evidence(a, p):
    gr, carb = a["gr_t"], a["carb_t"]
    p405_gr = _band_power_time(gr, 405.0)
    p405_c = _band_power_time(carb, 405.0)
    return {
        "rel60_gr": round(_band_power_time(gr, 60.0) / p405_gr, 4),
        "rel60_carb": round(_band_power_time(carb, 60.0) / p405_c, 4),
        "rel112_gr": round(_band_power_time(gr, 112.0) / p405_gr, 4),
        "rel112_carb": round(_band_power_time(carb, 112.0) / p405_c, 4),
    }


def _stable_seed(tag):
    return zlib.crc32(tag.encode("utf-8")) % 1_000_000




def _band_power(x, lo, hi, fs=2.0, win=101):
    x = x - np.convolve(x, np.ones(win) / win, mode="same")
    f, p = periodogram(x, fs=fs)
    wl = 1.0 / np.maximum(f, 1e-12)
    m = (wl >= lo) & (wl <= hi)
    return float(p[m].max()) if m.any() else 0.0


def instance_ok(out, truth):
    ev = truth["meta"]["q6_evidence"]
    mode = truth["meta"].get("q6_mode") or (
        "auto" if truth["Q6"]["p60_autogenic"] else
        ("allo" if truth["Q6"]["carb_has_60kyr_peak"] else "none"))
    if mode == "auto":
        ok = ev["rel60_gr"] > 0.5 and ev["rel60_carb"] < 0.1 and ev["rel112_gr"] > 0.5 and ev["rel112_carb"] < 0.1
    elif mode == "allo":
        ok = ev["rel60_gr"] > 0.5 and ev["rel60_carb"] > 0.5 and ev["rel112_gr"] > 0.5 and ev["rel112_carb"] > 0.5
    else:
        ok = ev["rel60_gr"] < 0.1 and ev["rel60_carb"] < 0.1
    return ok and float(out["A"]["depth"].max()) >= 260.0


def _dur(rng):
    while True:
        d = float(rng.integers(80, 381))
        if abs(d - 405) > 30 and abs(d - 810) > 30:
            return d


def sample_params(iid, seed):
    rng = np.random.default_rng(seed)
    q6_mode = str(rng.choice(["auto", "allo", "none"], p=[0.4, 0.4, 0.2]))
    q7_stronger = str(rng.choice(["GR", "CARB"]))
    if q7_stronger == "GR":
        gr_mod = float(np.round(rng.uniform(0.52, 0.70), 2))
        carb_mod = float(np.round(rng.uniform(0.30, 0.48), 2))
    else:
        gr_mod = float(np.round(rng.uniform(0.30, 0.48), 2))
        carb_mod = float(np.round(rng.uniform(0.52, 0.70), 2))
    T = float(np.round(rng.uniform(5.0, 7.8), 2))
    n_hiatus = int(rng.integers(3, 6))
    starts = []
    durs = []
    guard = 0
    while len(starts) < n_hiatus and guard < 100000:
        guard += 1
        t0 = rng.uniform(0.08 * T * 1000, 0.90 * T * 1000)
        d0 = _dur(rng)
        if all(abs(t0 - s) > max(600.0, d0 + 150.0) for s in starts):
            starts.append(float(t0))
            durs.append(d0)
    starts_arr = np.array(starts)
    durs_arr = np.array(durs)
    order = np.argsort(starts_arr)
    starts = starts_arr[order]
    durs = durs_arr[order]
    regional = int(rng.integers(0, len(starts)))
    b_local = sorted(rng.uniform(0.10 * T * 1000, 0.92 * T * 1000, int(rng.integers(1, 4))))
    return InstanceParams(
        iid=iid,
        seed=seed,
        T_myr=T,
        sr_base=float(np.round(rng.uniform(6.7, 8.6), 2)),
        sr_amp1=float(np.round(rng.uniform(0.3, 1.2), 2)),
        sr_amp2=float(np.round(rng.uniform(0.1, 0.5), 2)),
        sr_step_at=float(np.round(rng.uniform(0.3, 0.7) * T * 1000, 1)),
        sr_step=float(np.round(rng.uniform(-1.2, 1.2), 2)),
        obliq_kyr=float(np.round(rng.uniform(32.0, 36.5), 1)),
        prec_kyr=float(np.round(rng.uniform(18.0, 21.0), 1)),
        hiatus_times=[float(round(x, 1)) for x in starts],
        hiatus_durs=[float(x) for x in durs],
        regional_index=regional,
        b_local_times=[float(round(x, 1)) for x in b_local],
        b_local_durs=[_dur(rng) for _ in b_local],
        gr_mod=gr_mod,
        carb_mod=carb_mod,
        q6_mode=q6_mode,
        q7_stronger=q7_stronger,
        aut60=float(np.round(rng.uniform(0.24, 0.36), 2)),
        aut112=float(np.round(rng.uniform(0.17, 0.28), 2)),
        noise_gr=float(np.round(rng.uniform(0.08, 0.16), 3)),
        noise_carb=float(np.round(rng.uniform(0.10, 0.20), 3)),
        trend=float(np.round(rng.uniform(-0.4, 0.4), 2)),
    )


def write_instance(root, params, out, truth):
    d = os.path.join(root, "data", params.iid)
    os.makedirs(d, exist_ok=True)
    a, b = out["A"], out["B"]
    with open(os.path.join(d, "well_A.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("Depth_m,GR_API,CARB_pct\n")
        for depth, gr, carb in zip(a["depth"], a["gr"], a["carb"]):
            f.write(f"{depth:.1f},{gr:.4f},{carb:.4f}\n")
    with open(os.path.join(d, "well_B.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("Depth_m,GR_API\n")
        for depth, gr in zip(b["depth"], b["gr"]):
            f.write(f"{depth:.1f},{gr:.4f}\n")
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_v2")
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--test", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    truths, params_out = {}, {}
    for split, n in (("train", args.train), ("test", args.test)):
        for k in range(n):
            iid = f"{split}_{k:03d}"
            base = _stable_seed(iid)
            for attempt in range(25):
                params = sample_params(iid, seed=base + attempt * 7919)
                out, truth = make_instance(params)
                if instance_ok(out, truth):
                    break
            else:
                raise RuntimeError(f"could not generate a valid instance for {iid}")
            truth["meta"]["split"] = split
            truths[iid] = write_instance(args.out, params, out, truth)
            params_out[iid] = asdict(params)
            print(iid, "T=%.2f" % params.T_myr, "dA=%.0f" % truth["meta"]["depth_A_m"], "hiatus", len(params.hiatus_times))
    with open(os.path.join(args.out, "truths.json"), "w", encoding="utf-8") as f:
        json.dump(truths, f, indent=1)
    with open(os.path.join(args.out, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params_out, f, indent=1)
    print("wrote", len(truths), "instances to", args.out)


if __name__ == "__main__":
    main()
