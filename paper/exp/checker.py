# -*- coding: utf-8 -*-
"""Scoring for CycloBench-v2 (same schema as the original 8-question task)."""

import numpy as np

TOL = {"Q1": 0.10, "Q2": 0.050, "Q3_depth": 3.0, "Q3_dur": 50.0, "Q4": 0.20, "Q5": 2.0,
       "Q8_depth": 3.0, "Q8_dur": 50.0}


def _fail():
    return False, ""


def score_q1(a, t):
    v = a.get("Q1_total_duration_Myr")
    if not isinstance(v, (int, float)):
        return _fail()
    return abs(v - t["Q1_total_duration_Myr"]) <= TOL["Q1"], ""


def score_q2(a, t):
    v = a.get("Q2_age_250m_Ma")
    if not isinstance(v, (int, float)):
        return _fail()
    return abs(v - t["Q2_age_250m_Ma"]) <= TOL["Q2"], ""


def score_q3(a, t):
    hs = a.get("Q3_hiatuses")
    if not isinstance(hs, list) or len(hs) != len(t["Q3_hiatuses_depth_m"]):
        return _fail()
    used = [False] * len(hs)
    for d, k in zip(t["Q3_hiatuses_depth_m"], t["Q3_hiatuses_duration_kyr"]):
        hit = False
        for i, h in enumerate(hs):
            if used[i] or not isinstance(h, dict):
                continue
            if abs(h.get("depth_m", 1e9) - d) <= TOL["Q3_depth"] and abs(h.get("duration_kyr", 1e9) - k) <= TOL["Q3_dur"]:
                used[i] = True
                hit = True
                break
        if not hit:
            return _fail()
    return True, ""


def score_q4(a, t):
    v = a.get("Q4_mean_SR_cm_kyr")
    if not isinstance(v, (int, float)):
        return _fail()
    return abs(v - t["Q4_mean_SR_cm_kyr"]) <= TOL["Q4"], ""


def score_q5(a, t):
    v = a.get("Q5_obliquity_kyr")
    if not isinstance(v, (int, float)):
        return _fail()
    if abs(v - 41.0) < 1.0:
        return _fail()
    return abs(v - t["Q5_obliquity_kyr"]) <= TOL["Q5"], ""


def score_q6(a, t):
    q = a.get("Q6")
    if not isinstance(q, dict):
        return _fail()
    for k in ("p60_autogenic", "p112_autogenic", "carb_has_60kyr_peak", "carb_has_112kyr_peak"):
        if not isinstance(q.get(k), bool) or q[k] != t["Q6"][k]:
            return _fail()
    return True, ""


def score_q7(a, t):
    q = a.get("Q7")
    if not isinstance(q, dict):
        return _fail()
    if not isinstance(q.get("exists"), bool) or q["exists"] != t["Q7"]["exists"]:
        return _fail()
    if q.get("stronger_in") != t["Q7"]["stronger_in"]:
        return _fail()
    gf, cf = q.get("GR_factor"), q.get("CARB_factor")
    if not isinstance(gf, (int, float)) or not isinstance(cf, (int, float)):
        return _fail()
    if not (0.05 <= gf <= 1.5 and 0.05 <= cf <= 1.5) or cf <= gf:
        return _fail()
    return True, ""


def score_q8(a, t):
    q = a.get("Q8")
    if not isinstance(q, dict):
        return _fail()
    for key, tol in (("wellA_depth_m", "Q8_depth"), ("wellB_depth_m", "Q8_depth"), ("duration_kyr", "Q8_dur")):
        v = q.get(key)
        if not isinstance(v, (int, float)) or abs(v - t["Q8"][key]) > TOL[tol]:
            return _fail()
    return True, ""


SCORERS = [score_q1, score_q2, score_q3, score_q4, score_q5, score_q6, score_q7, score_q8]


def score(answers, truth):
    out = {}
    correct = 0
    for sc in SCORERS:
        ok, _ = sc(answers, truth)
        out[sc.__name__.replace("score_", "").upper()] = int(ok)
        correct += int(ok)
    out["total"] = correct / len(SCORERS)
    return out


def to_answer(truth):
    """Convert a stored truth record into the answer schema (tests / oracle)."""
    return {
        "Q1_total_duration_Myr": truth["Q1_total_duration_Myr"],
        "Q2_age_250m_Ma": truth["Q2_age_250m_Ma"],
        "Q3_hiatuses": [
            {"depth_m": d, "duration_kyr": k}
            for d, k in zip(truth["Q3_hiatuses_depth_m"], truth["Q3_hiatuses_duration_kyr"])
        ],
        "Q4_mean_SR_cm_kyr": truth["Q4_mean_SR_cm_kyr"],
        "Q5_obliquity_kyr": truth["Q5_obliquity_kyr"],
        "Q6": truth["Q6"],
        "Q7": truth["Q7"],
        "Q8": truth["Q8"],
    }

def _gauss(x, t, tol):
    try:
        return float(np.exp(-0.5 * ((float(x) - float(t)) / max(tol, 1e-9)) ** 2))
    except Exception:
        return 0.0


def _rel(x, t, rel=0.2):
    try:
        e = abs(float(x) - float(t)) / max(abs(float(t)), 1e-9)
        return float(1.0 / (1.0 + (e / rel) ** 2))
    except Exception:
        return 0.0


def soft_score(a, t):
    """Continuous 0-1 score per question (companion to the exact 0/1 score)."""
    out = {}
    out["Q1"] = _rel(a.get("Q1_total_duration_Myr"), t["Q1_total_duration_Myr"], 0.2)
    out["Q2"] = _rel(a.get("Q2_age_250m_Ma"), t["Q2_age_250m_Ma"], 0.0015)
    out["Q4"] = _rel(a.get("Q4_mean_SR_cm_kyr"), t["Q4_mean_SR_cm_kyr"], 0.2)
    out["Q5"] = _rel(a.get("Q5_obliquity_kyr"), t["Q5_obliquity_kyr"], 0.1)
    hs = a.get("Q3_hiatuses") if isinstance(a.get("Q3_hiatuses"), list) else []
    td, tk = t["Q3_hiatuses_depth_m"], t["Q3_hiatuses_duration_kyr"]
    matched, used = 0, set()
    for d, k in zip(td, tk):
        for i, h in enumerate(hs):
            if i in used or not isinstance(h, dict):
                continue
            if abs(h.get("depth_m", 1e9) - d) <= TOL["Q3_depth"] and abs(h.get("duration_kyr", 1e9) - k) <= TOL["Q3_dur"]:
                used.add(i); matched += 1; break
    prec = matched / max(len(hs), 1)
    rec = matched / max(len(td), 1)
    out["Q3"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    q6 = a.get("Q6") if isinstance(a.get("Q6"), dict) else {}
    keys = ("p60_autogenic", "p112_autogenic", "carb_has_60kyr_peak", "carb_has_112kyr_peak")
    out["Q6"] = sum(int(q6.get(k) == t["Q6"][k]) for k in keys) / 4.0
    q7 = a.get("Q7") if isinstance(a.get("Q7"), dict) else {}
    s7 = 0.5 * int(q7.get("exists") == t["Q7"]["exists"])
    s7 += 0.5 * int(q7.get("stronger_in") == t["Q7"]["stronger_in"])
    out["Q7"] = s7
    q8 = a.get("Q8") if isinstance(a.get("Q8"), dict) else {}
    out["Q8"] = (
        _rel(q8.get("wellA_depth_m"), t["Q8"]["wellA_depth_m"], 0.05)
        + _rel(q8.get("wellB_depth_m"), t["Q8"]["wellB_depth_m"], 0.05)
        + _rel(q8.get("duration_kyr"), t["Q8"]["duration_kyr"], 0.2)
    ) / 3.0
    out["total"] = sum(out.values()) / 8.0
    return out
