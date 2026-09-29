"""因子注册表：从 factor_list.xlsx 读取因子定义，生成唯一英文标识。

factor_list.xlsx 的列（按位置）：表名、序号、字段、因子名、类型、是否非空、
默认值、是否业务Key、备注。首列表头本身是 ".xlsx"（不规范），因此按中文名
匹配失败时回退到按位置取列。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REGISTRY_COLUMNS = ["factor_id", "name", "field", "table", "note"]

# 中文列名 -> 位置回退
_COL_SPECS = [
    ("表名", 0, "table"),
    ("字段", 2, "field"),
    ("因子名", 3, "name"),
    ("备注", 8, "note"),
]


def _get_column(df: pd.DataFrame, cn_name: str, pos: int) -> pd.Series:
    if cn_name in df.columns:
        return df[cn_name]
    return df.iloc[:, pos]


def _translate_to_english(name: str, note: str) -> str:
    """中文因子名 -> 简短英文标识的翻译钩子。

    当前 factor_list 中 101 个因子全是英文名，本函数实际不会触发。
    退化策略：从备注中提取 ASCII 字母数字片段拼接；取不到则用 "Factor"。
    将来如需真正的机器翻译，替换本函数即可。
    """
    parts = re.findall(r"[A-Za-z0-9]+", str(note))
    base = "_".join(parts)[:40].strip("_")
    return base or "Factor"


def make_factor_id(name: object, note: object, used: set[str]) -> str:
    """生成唯一英文标识：ASCII 名清洗后直接用；中文名走翻译钩子；重名加 _2/_3 后缀。"""
    s = str(name).strip()
    if s.isascii():
        base = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_") or "Factor"
    else:
        base = _translate_to_english(s, str(note))
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def build_registry(factor_list_path: str | Path) -> pd.DataFrame:
    """读取 factor_list.xlsx，返回注册表 DataFrame（列见 REGISTRY_COLUMNS）。"""
    df = pd.read_excel(factor_list_path)
    cols = {out: _get_column(df, cn, pos) for cn, pos, out in _COL_SPECS}
    reg = pd.DataFrame(cols)
    reg = reg.dropna(subset=["name", "field", "table"]).reset_index(drop=True)

    used: set[str] = set()
    reg["factor_id"] = [
        make_factor_id(n, nt, used) for n, nt in zip(reg["name"], reg["note"])
    ]
    reg["name"] = reg["name"].astype(str)
    reg["field"] = reg["field"].astype(str)
    reg["table"] = reg["table"].astype(str)
    reg["note"] = reg["note"].astype(str)
    return reg[REGISTRY_COLUMNS]


def save_registry(reg: pd.DataFrame, lib_dir: str | Path) -> Path:
    path = Path(lib_dir) / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：Excel 直接打开不乱码
    reg.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_registry(lib_dir: str | Path) -> pd.DataFrame:
    path = Path(lib_dir) / "registry.csv"
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
