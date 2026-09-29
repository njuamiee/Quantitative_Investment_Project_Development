"""北交所股票日线行情加载（前复权），供因子评价与绘图使用。

数据源：<项目根>/day-data/前复权/<6位代码>-<名称>.csv（GBK 编码），
列为 日期/股票代码/开盘/收盘/最高/最低/成交量/成交额/振幅/涨跌幅/涨跌额/换手率。

因子库 stockid（如 'BJ920010'）与文件名的映射：去掉 'BJ' 前缀后匹配
'<代码>-*.csv'。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _PKG_DIR.parent.parent  # 项目根 (data-kyts)

DEFAULT_DAILY_DIR = _ROOT_DIR / "day-data" / "前复权"

# 中文列名 -> 英文列名
_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def _find_file(stockid: str, daily_dir: Path) -> Path:
    code = stockid[2:] if stockid.upper().startswith("BJ") else stockid
    matches = sorted(daily_dir.glob(f"{code}-*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"未找到股票 {stockid!r} 的日线文件: {daily_dir}/{code}-*.csv"
        )
    return matches[0]


def load_daily_bars(
    stockid: str,
    daily_dir: str | Path = DEFAULT_DAILY_DIR,
) -> pd.DataFrame:
    """加载单只股票的日线行情（前复权）。

    返回 datetime 索引（date，升序）的 DataFrame，
    列为 open/high/low/close/volume/amount。
    """
    daily_dir = Path(daily_dir)
    path = _find_file(stockid, daily_dir)
    df = pd.read_csv(path, encoding="gbk")
    df = df.rename(columns=_COL_MAP)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
    return df[keep].astype(float)


def load_prices(
    stockids: list[str],
    daily_dir: str | Path = DEFAULT_DAILY_DIR,
) -> dict[str, pd.DataFrame]:
    """批量加载多只股票行情，返回 dict[stockid -> DataFrame]。

    个别股票文件缺失时告警并跳过（不中断）。
    """
    import warnings

    out: dict[str, pd.DataFrame] = {}
    for sid in stockids:
        try:
            out[sid] = load_daily_bars(sid, daily_dir)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
    return out
