"""数据加载。"""
from pathlib import Path
import pandas as pd

from .paths import data_root

SEEDS = list(range(8))


def list_seeds():
    """可用 seed 列表，如 [0, 1, ..., 7]。"""
    return list(SEEDS)


def load_well(seed, well="A"):
    """读取某 seed 的某口井数据。

    参数
    ----
    seed : int
        0..7
    well : str
        'A' 或 'B'（也接受 'well_A' / 'a' 等写法）

    返回
    ----
    pandas.DataFrame
        well A: Depth_m / GR_API / CARB_pct；well B: Depth_m / GR_API
    """
    w = str(well).upper().replace("WELL_", "").replace(".CSV", "")
    if w not in ("A", "B"):
        raise ValueError(f"well 只能是 'A' 或 'B'，收到 {well!r}")
    fp = Path(data_root()) / f"seed_{int(seed)}" / f"well_{w}.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"找不到数据文件 {fp}。请确认 seed 在 {SEEDS} 中，"
            f"或用环境变量 CYCLOBENCH_DATA 指定数据目录。"
        )
    return pd.read_csv(fp)


def load_seed(seed):
    """一次读入某 seed 的两口井：{'A': DataFrame, 'B': DataFrame}。"""
    return {"A": load_well(seed, "A"), "B": load_well(seed, "B")}
