"""因子库管理入口：FactorLibrary。

用法示例
--------
>>> from factors import FactorLibrary
>>> lib = FactorLibrary()          # 默认库目录 <项目根>/factor_lib
>>> lib.init()                     # 全量重建（约 10~20 分钟）
>>> lib.update()                   # 增量追加新日期的数据
>>> lib.list_factors()             # 所有因子的注册表
>>> lib.list_stocks()              # 所有股票代码
>>> s = lib.get("Alpha001", "BJ920010")          # 单股时间序列
>>> df = lib.get("Alpha001", start="2025-01-01") # 长表截面
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import builder, registry as registry_mod

_PKG_DIR = Path(__file__).resolve().parent          # src/factors
_SRC_DIR = _PKG_DIR.parent                          # src
_ROOT_DIR = _SRC_DIR.parent                         # 项目根 (data-kyts)

DEFAULT_LIB_DIR = _ROOT_DIR / "factor_lib"
DEFAULT_FACTOR_LIST = _SRC_DIR / "factor_list.xlsx"
DEFAULT_SRC_DATA = _ROOT_DIR / "factors_download" / "kyzqsh11"


class FactorLibrary:
    """北交所股票因子库。

    参数
    ----------
    lib_dir : 因子库（parquet + registry.csv）所在目录。
    factor_list_path : factor_list.xlsx 路径（init 时重建注册表用）。
    src_dir : 原始数据根目录（其下为 <表名>/<年份>/<YYYYMMDD>.zip）。
    prefix : 股票代码前缀过滤，默认 'BJ'（北交所）。
    """

    def __init__(
        self,
        lib_dir: str | Path = DEFAULT_LIB_DIR,
        factor_list_path: str | Path = DEFAULT_FACTOR_LIST,
        src_dir: str | Path = DEFAULT_SRC_DATA,
        prefix: str = "BJ",
    ) -> None:
        self.lib_dir = Path(lib_dir)
        self.factor_list_path = Path(factor_list_path)
        self.src_dir = Path(src_dir)
        self.prefix = prefix

    # ------------------------------------------------------------------
    # 注册表
    # ------------------------------------------------------------------
    def _load_registry(self) -> pd.DataFrame:
        """优先读库中的 registry.csv；不存在则从 factor_list.xlsx 重建并保存。"""
        reg_path = self.lib_dir / "registry.csv"
        if reg_path.exists():
            return registry_mod.load_registry(self.lib_dir)
        reg = registry_mod.build_registry(self.factor_list_path)
        registry_mod.save_registry(reg, self.lib_dir)
        return reg

    def list_factors(self) -> pd.DataFrame:
        """返回全部因子的注册表：factor_id / name / field / table / note。"""
        return self._load_registry()

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _table_fields(self, reg: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
        tf: dict[str, list[tuple[str, str]]] = {}
        for row in reg.itertuples(index=False):
            tf.setdefault(row.table, []).append((row.factor_id, row.field))
        return tf

    @staticmethod
    def _zip_progress(done: int, total: int, date: pd.Timestamp) -> None:
        print(f"[build] {done}/{total} 天, 当前日期 {date.date()}", flush=True)

    def init(self, progress=_zip_progress) -> None:
        """全量重建：从 factor_list.xlsx 重建注册表，重扫全部 zip 覆盖写出。"""
        reg = registry_mod.build_registry(self.factor_list_path)
        registry_mod.save_registry(reg, self.lib_dir)
        entries = builder.scan_zips(self.src_dir, reg["table"].unique().tolist())
        print(f"共扫描到 {len(entries)} 个日度数据文件", flush=True)
        accs, stock_ids = builder.accumulate(
            entries, self._table_fields(reg), self.prefix, progress
        )
        written = builder.write_factors(
            accs, stock_ids, self.lib_dir, merge_existing=False,
            progress=lambda k, n, fid: print(f"[write] {k}/{n} {fid}", flush=True),
        )
        print(f"init 完成：{len(written)} 个因子，{len(stock_ids)} 只股票", flush=True)

    def update(self, progress=_zip_progress) -> None:
        """增量更新：只处理晚于库中最大日期的新 zip，合并去重后重写各因子文件。"""
        reg = self._load_registry()
        table_fields = self._table_fields(reg)
        entries = builder.scan_zips(self.src_dir, list(table_fields))
        last = builder.max_dates(self.lib_dir, reg["factor_id"].tolist())
        built = [d for d in last.values() if d is not None]
        # 全局水位：各因子最大日期的最小值；早于等于它的文件一律跳过，
        # 个别因子如有重叠靠合并阶段去重兜底。
        watermark = min(built) if built else None
        if watermark is not None:
            entries = [e for e in entries if e[0] > watermark]
        if not entries:
            print("没有新数据，无需更新", flush=True)
            return
        print(f"发现 {len(entries)} 个新数据文件（自 {watermark} 之后）", flush=True)
        accs, stock_ids = builder.accumulate(entries, table_fields, self.prefix, progress)
        written = builder.write_factors(
            accs, stock_ids, self.lib_dir, merge_existing=True,
            progress=lambda k, n, fid: print(f"[write] {k}/{n} {fid}", flush=True),
        )
        print(f"update 完成：更新 {len(written)} 个因子", flush=True)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def _resolve_factor_id(self, factor: str) -> str:
        reg = self._load_registry()
        if factor in set(reg["factor_id"]):
            return factor
        hit = reg.loc[reg["name"] == factor, "factor_id"]
        if len(hit):
            return hit.iloc[0]
        raise KeyError(
            f"未知因子: {factor!r}，可用 list_factors() 查看全部 factor_id"
        )

    def _factor_path(self, factor: str) -> Path:
        fid = self._resolve_factor_id(factor)
        path = self.lib_dir / f"{fid}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"因子 {fid} 尚无数据文件: {path}")
        return path

    def list_stocks(self, factor: str | None = None) -> list[str]:
        """列出库中的股票代码；factor 缺省时用第一个因子文件推断。"""
        if factor is None:
            factor = self._load_registry()["factor_id"].iloc[0]
        path = self._factor_path(factor)
        return sorted(pd.read_parquet(path, columns=["stockid"])["stockid"].unique())

    def get(
        self,
        factor: str,
        stockid: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series | pd.DataFrame:
        """查询因子数据。

        stockid 给定时返回该股的时间序列（date 为索引的 Series，升序）；
        否则返回长表 DataFrame（date/stockid/value）。start/end 限定日期区间。
        factor 可用 factor_id 或原始因子名（如 'Alpha001'）。
        """
        df = pd.read_parquet(self._factor_path(factor))
        if stockid is not None:
            df = df[df["stockid"] == stockid]
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        if stockid is not None:
            s = df.set_index("date")["value"].sort_index()
            s.name = self._resolve_factor_id(factor)
            return s
        return df.sort_values(["date", "stockid"]).reset_index(drop=True)
