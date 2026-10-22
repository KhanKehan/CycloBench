# -*- coding: utf-8 -*-
"""Prompt builders for the three evaluation conditions.

The data are summarised (not pasted in full) because a well has ~1000 rows;
the tool-augmented condition instead exposes the CSV paths and a code tool.
"""

TASK = """You are a cyclostratigrapher. Two wells from the same Late Paleozoic lacustrine basin are provided.
Well A has Depth_m, GR_API, CARB_pct; well B has Depth_m, GR_API. Depth is in metres, sampled every 0.5 m.
The top of well A is 300.0 Ma and age increases downwards. Sedimentation rate may vary with depth and
hiatuses may be present in either well.

Definitions: Q1 is the total time spanned by well A from top to bottom, including time lost to hiatuses.
Q4 is the mean sedimentation rate, defined as total depth divided by that total time.

Answer these questions and return ONLY a JSON object with exactly these keys:
{
  "Q1_total_duration_Myr": float,
  "Q2_age_250m_Ma": float,
  "Q3_hiatuses": [{"depth_m": float, "duration_kyr": float}, ...],
  "Q4_mean_SR_cm_kyr": float,
  "Q5_obliquity_kyr": float,
  "Q6": {"p60_autogenic": bool, "p112_autogenic": bool,
         "carb_has_60kyr_peak": bool, "carb_has_112kyr_peak": bool},
  "Q7": {"exists": bool, "stronger_in": "GR" or "CARB", "GR_factor": float, "CARB_factor": float},
  "Q8": {"wellA_depth_m": float, "wellB_depth_m": float, "duration_kyr": float}
}
Q3 must list every hiatus in well A, in increasing depth. Q5 must account for the 300-Ma obliquity period
(do not answer 41 kyr). Q6 requires cross-validation of the 60-kyr and 112-kyr bands between GR and CARB.
Q7 asks whether ~170-kyr obliquity amplitude modulation exists and in which proxy it is stronger.
Q8 asks which hiatus is shared by both wells (the regional event)."""


def _summary(df, name):
    return (
        f"{name}: {len(df)} samples, depth {df.Depth_m.min():.1f}-{df.Depth_m.max():.1f} m, "
        f"mean step {(df.Depth_m.diff().dropna().mean()):.2f} m"
    )


def build_prompt(instance_dir, well_a, well_b, condition="zero_shot"):
    meta = _summary(well_a, "Well A") + "\n" + _summary(well_b, "Well B")
    if condition in ("zero_shot", "cot"):
        body = (
            f"{TASK}\n\nData summary:\n{meta}\n\n"
            "You do not have direct file access. Use the summary and your knowledge of cyclostratigraphy."
        )
        if condition == "cot":
            body += "\nThink step by step before the final JSON."
        return body
    if condition == "tool":
        return (
            f"{TASK}\n\n"
            "You have access to a restricted Python sandbox: A and B are pandas DataFrames "
            "already loaded (depth and proxies), and np, pd, signal (scipy.signal) and "
            "stats (scipy.stats) are available. No files, no network, no imports. "
            "Write code in a fenced python block; its printed output is returned to you. "
            "After at most three code turns, return the final JSON."
        )
    raise ValueError(condition)
