"""CycloBench —— 天文旋回地层学可复现基准。

用法::

    import cyclobench as cb

    cb.list_seeds()                       # [0, 1, ..., 7]
    wa = cb.load_well(seed=0, well='A')   # pandas.DataFrame: Depth_m / GR_API / CARB_pct
    res = cb.evaluate('answers.json', seed=0)
    print(res['reward'], res['correct'], '/ 8')
"""
from .data import SEEDS, list_seeds, load_well, load_seed
from .evaluate import evaluate, load_truths, TOL

__version__ = "0.1.0"
__all__ = ["SEEDS", "list_seeds", "load_well", "load_seed",
           "evaluate", "load_truths", "TOL"]
