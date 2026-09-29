"""因子库构建器：日期外层循环 + 内存累积 + 末尾一次性写出。

循环方案评估（详见实施计划）：
- 因子外层 × 股票内层：每个因子重扫全部 zip，约 101 × 2430 ≈ 24.6 万次解压，最差。
- 日期外层、每日立即落盘：zip 只解压一次，但全量初始化需 24.6 万次 parquet
  读-改-写，I/O 爆炸。
- 本模块采用：日期外层，每天读入 zip 后立即按前缀过滤股票（BJ 只占约 5%），
  把 (date, stockid, value) 累积进内存；全部日期扫完后每因子排序并一次性写出。
  总计 2430 次解压 + 101 次文件写，内存约 1GB（float64），64 位环境无压力。

增量更新复用同一流程：调用方先按库中已有最大日期过滤 zip 列表，此处产出的
增量数据与旧文件 concat 后按 (stockid, date) 去重重写，天然幂等。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATE_ZIP_RE = re.compile(r"^(\d{8})\.zip$")

# zip_entries: (date: pd.Timestamp, table: str, path: Path)
ZipEntry = tuple


def scan_zips(src_dir: str | Path, tables: list[str]) -> list[ZipEntry]:
    """扫描 src_dir/<表名>/<年份>/<YYYYMMDD>.zip，返回按日期升序的条目列表。

    年份目录下混有的已解压同名子目录（如 20251231/）会被正则说明自动忽略。
    """
    src_dir = Path(src_dir)
    entries: list[ZipEntry] = []
    for table in sorted(set(tables)):
        table_dir = src_dir / table
        if not table_dir.is_dir():
            warnings.warn(f"表目录不存在，跳过: {table_dir}")
            continue
        for year_dir in sorted(table_dir.iterdir()):
            if not (year_dir.is_dir() and year_dir.name.isdigit()):
                continue
            for f in sorted(year_dir.iterdir()):
                m = DATE_ZIP_RE.match(f.name)
                if m:
                    entries.append((pd.Timestamp(m.group(1)), table, f))
    entries.sort(key=lambda e: e[0])
    return entries


class _Accumulator:
    """单个因子的增量数据累积器（按天追加 numpy 数组块）。"""

    __slots__ = ("dates", "codes", "vals")

    def __init__(self) -> None:
        self.dates: list[np.ndarray] = []  # int32, yyyymmdd
        self.codes: list[np.ndarray] = []  # int32, 全局 stockid -> code
        self.vals: list[np.ndarray] = []  # float64

    def append(self, date_int: int, codes: np.ndarray, vals: np.ndarray) -> None:
        self.dates.append(np.full(len(codes), date_int, dtype=np.int32))
        self.codes.append(codes)
        self.vals.append(vals)

    def to_frame(self, stock_ids: list[str]) -> pd.DataFrame:
        dates = np.concatenate(self.dates)
        codes = np.concatenate(self.codes)
        vals = np.concatenate(self.vals)
        stockid = np.asarray(stock_ids, dtype=object)[codes]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                "stockid": stockid,
                "value": vals,
            }
        )


def accumulate(
    entries: list[ZipEntry],
    table_fields: dict[str, list[tuple[str, str]]],
    prefix: str = "BJ",
    progress=None,
) -> tuple[dict[str, _Accumulator], list[str]]:
    """日期外层循环：逐日解压 zip，按前缀过滤股票，按因子累积 (date, code, value)。

    参数
    ----------
    entries : scan_zips 的输出（已按日期升序）。
    table_fields : 表名 -> [(factor_id, 字段列名), ...]。
    prefix : 股票代码前缀过滤（北交所为 'BJ'）。
    progress : 可选回调，每处理 50 天调用 progress(已完成数, 总数, 日期)。

    返回
    -------
    (factor_id -> _Accumulator, 全局 stockid 列表)
    """
    accs: dict[str, _Accumulator] = {
        fid: _Accumulator() for pairs in table_fields.values() for fid, _ in pairs
    }
    stock_code: dict[str, int] = {}
    stock_ids: list[str] = []
    total = len(entries)

    for i, (date, table, path) in enumerate(entries):
        pairs = table_fields.get(table)
        if pairs:
            try:
                df = pd.read_csv(path, compression="zip")
            except Exception as exc:  # zip 损坏等：跳过该日
                warnings.warn(f"读取失败，跳过 {path}: {exc}")
                continue
            stockid = df["stockid"].astype(str)
            mask = stockid.str.startswith(prefix).to_numpy()
            if mask.any():
                day_stocks = stockid[mask].to_numpy()
                codes = np.empty(len(day_stocks), dtype=np.int32)
                for j, s in enumerate(day_stocks):
                    c = stock_code.get(s)
                    if c is None:
                        c = len(stock_ids)
                        stock_code[s] = c
                        stock_ids.append(s)
                    codes[j] = c
                date_int = date.year * 10000 + date.month * 100 + date.day
                for fid, field in pairs:
                    if field not in df.columns:
                        continue  # 该日缺此因子列：跳过
                    vals = pd.to_numeric(
                        df.loc[mask, field], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    accs[fid].append(date_int, codes, vals)
        if progress and ((i + 1) % 50 == 0 or i + 1 == total):
            progress(i + 1, total, date)
    return accs, stock_ids


def max_dates(lib_dir: str | Path, factor_ids: list[str]) -> dict[str, pd.Timestamp | None]:
    """各因子库文件中已有数据的最大日期；文件不存在或为空则为 None。"""
    lib_dir = Path(lib_dir)
    result: dict[str, pd.Timestamp | None] = {}
    for fid in factor_ids:
        path = lib_dir / f"{fid}.parquet"
        if path.exists():
            dates = pd.read_parquet(path, columns=["date"])["date"]
            result[fid] = dates.max() if len(dates) else None
        else:
            result[fid] = None
    return result


def write_factors(
    accs: dict[str, _Accumulator],
    stock_ids: list[str],
    lib_dir: str | Path,
    merge_existing: bool,
    progress=None,
) -> list[Path]:
    """把累积结果写为每因子一个 parquet（date/stockid/value，按 stockid,date 排序）。

    merge_existing=True 时先读旧文件 concat，按 (stockid, date) 去重（新数据优先）
    再重写——用于增量更新，幂等；False 时直接覆盖写——用于全量重建。
    """
    lib_dir = Path(lib_dir)
    lib_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    items = [(fid, acc) for fid, acc in accs.items() if acc.dates]
    for k, (fid, acc) in enumerate(items):
        frame = acc.to_frame(stock_ids)
        path = lib_dir / f"{fid}.parquet"
        if merge_existing and path.exists():
            old = pd.read_parquet(path)
            frame = pd.concat([old, frame], ignore_index=True)
        frame = (
            frame.drop_duplicates(subset=["stockid", "date"], keep="last")
            .sort_values(["stockid", "date"])
            .reset_index(drop=True)
        )
        frame.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        written.append(path)
        if progress:
            progress(k + 1, len(items), fid)
    return written
