# -*- coding: utf-8 -*-
"""
公平性自检：确认每个 seed 的数据中：
1) 405 kyr 长偏心率信号在 GR 谱中为显著主峰（可调谐）；
2) 自生 ~60 kyr 与 ~112 kyr 峰存在于 GR、缺席于 CARB（Q6 可判）；
3) 斜率信号在 CARB 中清晰（Q5/Q7 可解）。
输出公平性报告 JSON（验收材料用）。
"""
import numpy as np
import pandas as pd
import json
import os
from scipy.signal import periodogram

BASE = r'C:\Users\27218\Desktop\Harbor_旋回地层学任务包'
report = {}

for k in range(8):
    p = os.path.join(BASE, 'data', f'seed_{k}')
    A = pd.read_csv(os.path.join(p, 'well_A.csv'))
    d = A['Depth_m'].values

    def top_peaks(vals, n=6):
        v = vals - np.polyval(np.polyfit(d, vals, 3), d)
        f, P = periodogram(v, fs=2.0)
        f, P = f[1:], P[1:]
        wl = 1.0 / f
        m = wl < 300
        idx = np.argsort(P[m])[::-1][:n]
        return [round(float(wl[m][i]), 1) for i in idx]

    gr_peaks = top_peaks(A['GR_API'].values)
    ca_peaks = top_peaks(A['CARB_pct'].values)

    def has_peak(peaks, lo, hi):
        return any(lo <= p <= hi for p in peaks)

    rep = {
        'GR_top6_wavelengths_m': gr_peaks,
        'CARB_top6_wavelengths_m': ca_peaks,
        'GR_405band_present(20-45m)': has_peak(gr_peaks, 20, 45),
        'GR_60kyr_autogenic_present(3.5-5m)': has_peak(gr_peaks, 3.5, 5.0),
        'GR_112kyr_autogenic_present(7.5-10m)': has_peak(gr_peaks, 7.5, 10.0),
        'CARB_obliquity_present(1.8-3.5m)': has_peak(ca_peaks, 1.8, 3.5),
        'CARB_lacks_60kyr_peak': not has_peak(ca_peaks, 3.5, 5.0),
        'CARB_lacks_112kyr_peak': not has_peak(ca_peaks, 7.5, 10.0),
    }
    report[f'seed_{k}'] = rep
    ok = rep['GR_405band_present(20-45m)'] and rep['CARB_obliquity_present(1.8-3.5m)']
    print(f'seed {k}: GR={gr_peaks} CARB={ca_peaks} 核心信号可解: {ok}')

out = os.path.join(BASE, 'solution', 'fairness_report.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('written:', out)
