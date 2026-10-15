#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harbor 判分脚本：旋回地层学双井任务（8 题，逐项 0/1，reward = 正确项数 / 8）
用法:
  python3 check_answers.py <answers.json> <truths.json> <seed_key> [reward_file]
"""
import json
import sys
import os

TOL = {
    'Q1': 0.10,      # Myr
    'Q2': 0.050,     # Ma
    'Q3_depth': 3.0,   # m
    'Q3_dur': 50.0,    # kyr
    'Q4': 0.20,      # cm/kyr
    'Q5': 2.0,       # kyr
    'Q8_depth': 3.0,   # m
    'Q8_dur': 50.0,    # kyr
}

def fail(msg):
    return False, msg

def ok():
    return True, ''

def score_q1(ans, t):
    v = ans.get('Q1_total_duration_Myr')
    if not isinstance(v, (int, float)):
        return fail('Q1 缺失或非数值')
    return ok() if abs(v - t['Q1_total_duration_Myr']) <= TOL['Q1'] else fail(f'Q1 超差: {v} vs {t["Q1_total_duration_Myr"]}')

def score_q2(ans, t):
    v = ans.get('Q2_age_250m_Ma')
    if not isinstance(v, (int, float)):
        return fail('Q2 缺失或非数值')
    return ok() if abs(v - t['Q2_age_250m_Ma']) <= TOL['Q2'] else fail(f'Q2 超差: {v} vs {t["Q2_age_250m_Ma"]}')

def score_q3(ans, t):
    hs = ans.get('Q3_hiatuses')
    if not isinstance(hs, list):
        return fail('Q3 格式错误')
    td = t['Q3_hiatuses_depth_m']
    tdur = t['Q3_hiatuses_duration_kyr']
    if len(hs) != len(td):
        return fail(f'Q3 间断数量错误: {len(hs)} vs {len(td)}（不得多报/漏报）')
    used = [False] * len(hs)
    for depth_t, dur_t in zip(td, tdur):
        matched = False
        for i, h in enumerate(hs):
            if used[i]:
                continue
            if not isinstance(h, dict) or 'depth_m' not in h or 'duration_kyr' not in h:
                return fail('Q3 元素格式错误')
            if abs(h['depth_m'] - depth_t) <= TOL['Q3_depth'] and abs(h['duration_kyr'] - dur_t) <= TOL['Q3_dur']:
                used[i] = True
                matched = True
                break
        if not matched:
            return fail(f'Q3 未找到间断 深度{depth_t}m/{dur_t}kyr')
    return ok()

def score_q4(ans, t):
    v = ans.get('Q4_mean_SR_cm_kyr')
    if not isinstance(v, (int, float)):
        return fail('Q4 缺失或非数值')
    return ok() if abs(v - t['Q4_mean_SR_cm_kyr']) <= TOL['Q4'] else fail(f'Q4 超差: {v} vs {t["Q4_mean_SR_cm_kyr"]}')

def score_q5(ans, t):
    v = ans.get('Q5_obliquity_kyr')
    if not isinstance(v, (int, float)):
        return fail('Q5 缺失或非数值')
    if abs(v - 41.0) < 1.0:
        return fail('Q5 使用现代斜率周期 41 kyr，未做时代校正')
    return ok() if abs(v - t['Q5_obliquity_kyr']) <= TOL['Q5'] else fail(f'Q5 超差: {v} vs {t["Q5_obliquity_kyr"]}')

def score_q6(ans, t):
    q = ans.get('Q6')
    if not isinstance(q, dict):
        return fail('Q6 格式错误')
    tt = t['Q6']
    for key in ['p60_autogenic', 'p112_autogenic', 'carb_has_60kyr_peak', 'carb_has_112kyr_peak']:
        if not isinstance(q.get(key), bool):
            return fail(f'Q6.{key} 缺失或非布尔')
        if q[key] != tt[key]:
            return fail(f'Q6.{key} 错误: {q[key]} vs {tt[key]}')
    return ok()

def score_q7(ans, t):
    q = ans.get('Q7')
    if not isinstance(q, dict):
        return fail('Q7 格式错误')
    tt = t['Q7']
    if not isinstance(q.get('exists'), bool) or q['exists'] != tt['exists']:
        return fail('Q7.exists 错误')
    if q.get('stronger_in') != tt['stronger_in']:
        return fail(f'Q7.stronger_in 错误: {q.get("stronger_in")} vs {tt["stronger_in"]}')
    gf = q.get('GR_factor'); cf = q.get('CARB_factor')
    if not isinstance(gf, (int, float)) or not isinstance(cf, (int, float)):
        return fail('Q7 因子缺失或非数值')
    if not (0.05 <= gf <= 1.5 and 0.05 <= cf <= 1.5):
        return fail('Q7 因子超出合理范围 [0.05,1.5]')
    if cf <= gf:
        return fail(f'Q7 CARB_factor 应大于 GR_factor: {cf} vs {gf}')
    return ok()

def score_q8(ans, t):
    q = ans.get('Q8')
    if not isinstance(q, dict):
        return fail('Q8 格式错误')
    tt = t['Q8']
    checks = [
        ('wellA_depth_m', 'Q8_depth'), ('wellB_depth_m', 'Q8_depth'), ('duration_kyr', 'Q8_dur'),
    ]
    for key, tol_key in checks:
        v = q.get(key)
        if not isinstance(v, (int, float)):
            return fail(f'Q8.{key} 缺失或非数值')
        if abs(v - tt[key]) > TOL[tol_key]:
            return fail(f'Q8.{key} 超差: {v} vs {tt[key]}')
    return ok()

def main():
    answers_path, truths_path, seed_key = sys.argv[1], sys.argv[2], sys.argv[3]
    reward_file = sys.argv[4] if len(sys.argv) > 4 else None

    with open(truths_path, encoding='utf-8') as f:
        truths = json.load(f)
    if seed_key not in truths:
        print('ERROR: unknown seed', seed_key, file=sys.stderr)
        sys.exit(2)
    t = truths[seed_key]

    if not os.path.exists(answers_path):
        reward, results = 0.0, ['answers.json 不存在'] * 8
    else:
        try:
            with open(answers_path, encoding='utf-8') as f:
                ans = json.load(f)
        except Exception as e:
            reward, results = 0.0, [f'answers.json 解析失败: {e}'] * 8
            if reward_file:
                with open(reward_file, 'w') as f:
                    f.write('0.0')
            print(json.dumps({'seed': seed_key, 'reward': reward, 'details': results}, ensure_ascii=False))
            sys.exit(0)

        scorers = [score_q1, score_q2, score_q3, score_q4, score_q5, score_q6, score_q7, score_q8]
        results = []
        correct = 0
        for sc in scorers:
            passed, msg = sc(ans, t)
            results.append({'item': sc.__name__.replace('score_', 'Q'), 'pass': passed, 'detail': msg})
            if passed:
                correct += 1
        reward = round(correct / 8.0, 3)

    if reward_file:
        with open(reward_file, 'w') as f:
            f.write(str(reward))
    print(json.dumps({'seed': seed_key, 'reward': reward, 'correct': correct, 'details': results}, ensure_ascii=False))

if __name__ == '__main__':
    main()
