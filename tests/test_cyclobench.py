# -*- coding: utf-8 -*-
"""CycloBench 基础测试（不依赖 R，纯 Python）。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import cyclobench as cb

ROOT = Path(__file__).resolve().parent.parent


def _perfect_answer(t):
    """由真值构造满分答案。"""
    return {
        "Q1_total_duration_Myr": t["Q1_total_duration_Myr"],
        "Q2_age_250m_Ma": t["Q2_age_250m_Ma"],
        "Q3_hiatuses": [
            {"depth_m": d, "duration_kyr": k}
            for d, k in zip(t["Q3_hiatuses_depth_m"], t["Q3_hiatuses_duration_kyr"])
        ],
        "Q4_mean_SR_cm_kyr": t["Q4_mean_SR_cm_kyr"],
        "Q5_obliquity_kyr": t["Q5_obliquity_kyr"],
        "Q6": t["Q6"],
        "Q7": {k: t["Q7"][k] for k in ["exists", "stronger_in", "GR_factor", "CARB_factor"]},
        "Q8": t["Q8"],
    }


def test_list_seeds():
    assert cb.list_seeds() == list(range(8))


def test_load_well_shapes_and_columns():
    wa = cb.load_well(0, "A")
    assert list(wa.columns) == ["Depth_m", "GR_API", "CARB_pct"]
    assert len(wa) > 100
    wb = cb.load_well(0, "B")
    assert list(wb.columns) == ["Depth_m", "GR_API"]
    assert len(wb) > 100


def test_load_well_bad_input():
    with pytest.raises(ValueError):
        cb.load_well(0, "C")


@pytest.mark.parametrize("k", range(8))
def test_perfect_answer_scores_full(k):
    truths = cb.load_truths()
    res = cb.evaluate(_perfect_answer(truths[f"seed_{k}"]), seed=k)
    assert res["reward"] == 1.0
    assert res["correct"] == 8


def test_empty_answer_scores_zero():
    res = cb.evaluate({}, seed=0)
    assert res["reward"] == 0.0
    assert res["correct"] == 0


def test_q5_rejects_modern_obliquity():
    """答现代 41 kyr 应判 0（未做时代校正）。"""
    truths = cb.load_truths()
    ans = _perfect_answer(truths["seed_0"])
    ans["Q5_obliquity_kyr"] = 41.0
    res = cb.evaluate(ans, seed=0)
    q5 = [d for d in res["details"] if d["item"] == "Q5"][0]
    assert q5["pass"] is False


def test_check_answers_script(tmp_path):
    """原版判分脚本也应给出满分（含 Q6 键名一致性）。"""
    truths = cb.load_truths()
    ans_path = tmp_path / "answers.json"
    ans_path.write_text(json.dumps(_perfect_answer(truths["seed_0"]), ensure_ascii=False),
                        encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "check_answers.py"),
         str(ans_path), str(ROOT / "tests" / "truths.json"), "seed_0"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["reward"] == 1.0
