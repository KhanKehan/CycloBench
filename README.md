# CycloBench

[![CI](https://github.com/KhanKehan/CycloBench/actions/workflows/ci.yml/badge.svg)](https://github.com/KhanKehan/CycloBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pyproject.toml)
[![Task](https://img.shields.io/badge/Harbor-task-8A2BE2.svg)](task.toml)

> **天文旋回地层学的可复现 AI 基准**
> A reproducible, quantitatively-verifiable AI benchmark for astrochronology & cyclostratigraphy.

CycloBench 考察模型/程序能否在**陆相湖盆地层记录**中完成完整的旋回地层学分析链：
**天文信号检测 → 沉积速率估计 → 405 kyr 调谐 → 沉积间断识别 → 多指标交叉验证 → 多井对比**。
每个任务实例给出一口/两口井的测井记录（自然伽马 GR、碳酸盐含量 CARB），要求回答 8 道**数值可验证**的题。

本基准的难度来自**真实科学瓶颈**：陆相记录天文信号弱、沉积速率非平稳、存在多处沉积间断与强自生红噪声——
任何单步简化（默认 FFT 置信、单一沉积速率、忽略间断、单指标结论）都会导致至少一题超差。

---

## 任务概览

- **井 A**：`Depth_m` / `GR_API` / `CARB_pct`
- **井 B**：`Depth_m` / `GR_API`
- 锚点：井 A 顶部（0 m）= **300.0 Ma**，年龄向下增大
- 输出：`answers.json`（字段固定）+ 过程图 `figures/*.png`

| 题 | 内容 | 容差 |
|---|---|---|
| Q1 | 总沉积时长（Myr） | ±0.10 |
| Q2 | 250 m 处绝对年龄（Ma） | ±0.050 |
| Q3 | 全部沉积间断（深度 + 缺失时间） | 深度 ±3 m；时长 ±50 kyr |
| Q4 | 平均沉积速率（cm/kyr） | ±0.20 |
| Q5 | 斜率主周期（kyr，**需时代校正**） | ±2.0 |
| Q6 | ~60/~112 kyr 峰的成因 + CARB 交叉验证 | 全部一致 |
| Q7 | ~170 kyr 斜率振幅调制（存在性 + 更强指标 + 因子） | 方向 + 范围一致 |
| Q8 | 两井共有的区域性间断 | 深度 ±3 m；时长 ±50 kyr |

评测：8 项逐项 0/1，`reward = 正确项数 / 8`。

---

## 数据说明

- **合成数据**，但具科学依据（非随机造数）：
  - 天文信号按 **300 Ma 天文解**（La2004 外推 + Waltham 2015）合成：405/125/95 kyr 偏心率、35 kyr 斜率（含 ~170 kyr 振幅调制）、21/18 kyr 岁差；
  - 噪声为**非平稳 AR(1) 红噪声**（ρ=0.92，σ=2.2→3.5）；
  - 叠加两组**自生准周期**（~58–63 kyr、~112–118 kyr），模拟三角洲朵体迁移等局部过程；
  - **分段沉积速率**、**多处沉积间断**（井 A 五处、井 B 两处，其一为两井共有的区域事件）；
  - 深度域按 0.5 m 均匀重采样。
- **8 个 seed**（`seed_0..7`）真值互异（seed-evolving），防答案记忆。
- 全部数据由本任务方自行生成，无第三方版权，可自由用于训练与评测。

详见 `docs/任务说明.md`。

---

## 仓库结构

```
CycloBench/
├── README.md
├── LICENSE
├── instruction.md              # 题面（8 问 + 输出格式）
├── task.toml                   # 任务配置（资源/超时/网络）
├── docs/任务说明.md            # 背景、生成机制、难度证据、Rubric
├── data/seed_0..7/            # well_A.csv, well_B.csv（8 个 seed）
├── tests/
│   ├── check_answers.py       # 判分脚本（0/1 + 容差）
│   ├── truths.json            # 各 seed 真值
│   └── test.sh
├── solution/
│   ├── solve.sh / solve.R     # R + astrochron 参考解
│   ├── verify_fairness.py     # 公平性自检
│   └── fairness_report.json   # 8 seed 公平性报告
├── environment/Dockerfile     # R + astrochron 环境
├── cyclobench/                # Python 包（数据加载 + 评测）
└── pyproject.toml
```

---

## 快速开始

```bash
pip install -e .
```

```python
import cyclobench as cb

cb.list_seeds()                     # [0, 1, ..., 7]
wa = cb.load_well(seed=0, well='A') # pandas.DataFrame: Depth_m / GR_API / CARB_pct
wb = cb.load_well(seed=0, well='B')

# 评测你的答案（answers.json 或 dict）
res = cb.evaluate('answers.json', seed=0)
print(res['reward'], res['correct'], '/ 8')
```

命令行判分：

```bash
python tests/check_answers.py answers.json tests/truths.json seed_0
```

参考解（R）：

```bash
bash solution/solve.sh        # 依赖 R 4.4+ 与 astrochron
```

---

## 评测与公平性

- 判分只读 `answers.json`，与参考答案隔离；
- 参考解（`solution/solve.R`）按标准流程应得满分；
- 每个 seed 均通过 **AR(1) 红噪声蒙特卡洛检验**：405 kyr、~60 kyr、~112 kyr 信号显著高于 95% 置信线，
  而 CARB 在 57–63 / 108–118 kyr 波段不显著（保证 Q6 可判）。见 `solution/fairness_report.json`。

---

## 引用

如果你使用了本基准，请引用：

```bibtex
@misc{cyclobench2026,
  title  = {CycloBench: A Reproducible Benchmark for Cyclostratigraphic Analysis},
  author = {Yu, Kehan},
  year   = {2026},
  note   = {https://github.com/KhanKehan/CycloBench}
}
```

## 许可证

代码与数据以 **MIT License** 发布，见 `LICENSE`。
