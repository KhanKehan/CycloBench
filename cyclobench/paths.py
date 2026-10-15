"""路径解析：数据与真值的位置。

优先读环境变量，其次按仓库布局解析（repo_root/data 与 repo_root/tests）。
"""
import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    env = os.environ.get("CYCLOBENCH_DATA")
    if env:
        return Path(env)
    return repo_root() / "data"


def truths_path() -> Path:
    env = os.environ.get("CYCLOBENCH_TRUTHS")
    if env:
        return Path(env)
    return repo_root() / "tests" / "truths.json"
