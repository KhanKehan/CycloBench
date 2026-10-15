"""评测：把 ``answers.json`` 与各 seed 真值逐题比对（8 项 0/1，reward = 正确数 / 8）。

与 ``tests/check_answers.py`` 保持一致的容差与判分逻辑。
"""
import json
from pathlib import Path

from .paths import truths_path

TOL = {
    "Q1": 0.10,       # Myr
    "Q2": 0.050,      # Ma
    "Q3_depth": 3.0,  # m
    "Q3_dur": 50.0,   # kyr
    "Q4": 0.20,       # cm/kyr
    "Q5": 2.0,        # kyr
    "Q8_depth": 3.0,  # m
    "Q8_dur": 50.0,   # kyr
}


def load_truths(path=None):
    """读取真值字典（默认 ``tests/truths.json``）。"""
    p = Path(path) if path else truths_path()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _seed_key(seed):
    if isinstance(seed, str):
        return seed if seed.startswith("seed_") else f"seed_{seed}"
    return f"seed_{int(seed)}"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _score_q1(ans, t):
    v = ans.get("Q1_total_duration_Myr")
    return _num(v) and abs(v - t["Q1_total_duration_Myr"]) <= TOL["Q1"]


def _score_q2(ans, t):
    v = ans.get("Q2_age_250m_Ma")
    return _num(v) and abs(v - t["Q2_age_250m_Ma"]) <= TOL["Q2"]


def _score_q3(ans, t):
    hs = ans.get("Q3_hiatuses")
    if not isinstance(hs, list):
        return False
    td = t["Q3_hiatuses_depth_m"]
    tdur = t["Q3_hiatuses_duration_kyr"]
    if len(hs) != len(td):
        return False
    used = [False] * len(hs)
    for depth_t, dur_t in zip(td, tdur):
        matched = False
        for i, h in enumerate(hs):
            if used[i] or not isinstance(h, dict):
                continue
            if "depth_m" not in h or "duration_kyr" not in h:
                return False
            if (abs(h["depth_m"] - depth_t) <= TOL["Q3_depth"]
                    and abs(h["duration_kyr"] - dur_t) <= TOL["Q3_dur"]):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _score_q4(ans, t):
    v = ans.get("Q4_mean_SR_cm_kyr")
    return _num(v) and abs(v - t["Q4_mean_SR_cm_kyr"]) <= TOL["Q4"]


def _score_q5(ans, t):
    v = ans.get("Q5_obliquity_kyr")
    if not _num(v):
        return False
    if abs(v - 41.0) < 1.0:
        return False  # 未做时代校正的典型错误
    return abs(v - t["Q5_obliquity_kyr"]) <= TOL["Q5"]


def _score_q6(ans, t):
    q = ans.get("Q6")
    if not isinstance(q, dict):
        return False
    tt = t["Q6"]
    for key in ["p60_autogenic", "p112_autogenic",
                "carb_has_60kyr_peak", "carb_has_112kyr_peak"]:
        if not isinstance(q.get(key), bool) or q[key] != tt[key]:
            return False
    return True


def _score_q7(ans, t):
    q = ans.get("Q7")
    if not isinstance(q, dict):
        return False
    tt = t["Q7"]
    if not isinstance(q.get("exists"), bool) or q["exists"] != tt["exists"]:
        return False
    if q.get("stronger_in") != tt["stronger_in"]:
        return False
    gf, cf = q.get("GR_factor"), q.get("CARB_factor")
    if not _num(gf) or not _num(cf):
        return False
    if not (0.05 <= gf <= 1.5 and 0.05 <= cf <= 1.5):
        return False
    return cf > gf


def _score_q8(ans, t):
    q = ans.get("Q8")
    if not isinstance(q, dict):
        return False
    tt = t["Q8"]
    for key, tol_key in [("wellA_depth_m", "Q8_depth"),
                         ("wellB_depth_m", "Q8_depth"),
                         ("duration_kyr", "Q8_dur")]:
        v = q.get(key)
        if not _num(v) or abs(v - tt[key]) > TOL[tol_key]:
            return False
    return True


_SCORERS = [_score_q1, _score_q2, _score_q3, _score_q4,
            _score_q5, _score_q6, _score_q7, _score_q8]
_ITEM_NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]


def evaluate(answers, seed=0, truths=None, truths_file=None):
    """评测一组答案。

    参数
    ----
    answers : dict | str | Path
        答案字典，或 ``answers.json`` 路径。
    seed : int | str
        0..7，或 'seed_0' 之类。
    truths : dict, 可选
        真值字典；不给则从 ``tests/truths.json`` 读取。

    返回
    ----
    dict
        {'seed', 'reward', 'correct', 'details': [...]}
    """
    key = _seed_key(seed)
    if truths is None:
        truths = load_truths(truths_file)
    if key not in truths:
        raise KeyError(f"未知 seed: {key}（真值里有 {list(truths.keys())}）")
    t = truths[key]

    if isinstance(answers, (str, Path)):
        p = Path(answers)
        if not p.exists():
            return {"seed": key, "reward": 0.0, "correct": 0,
                    "details": ["answers.json 不存在"] * 8}
        with open(p, encoding="utf-8") as f:
            ans = json.load(f)
    else:
        ans = answers

    details, correct = [], 0
    for name, sc in zip(_ITEM_NAMES, _SCORERS):
        passed = bool(sc(ans, t))
        details.append({"item": name, "pass": passed})
        correct += passed
    return {"seed": key, "reward": round(correct / 8.0, 3),
            "correct": correct, "details": details}
