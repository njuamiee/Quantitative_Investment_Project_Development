"""北交所股票因子库管理系统 + 因子评价模块。

因子库（registry/builder/library/price_data）：管理 <表名>/<年份>/<日期>.zip
原始因子数据，构建为每因子一个 parquet（date/stockid/value 长表）的因子库，
并提供查询与增量更新。

因子评价（eval_config/stats_utils/evaluator/report，移植自 A 工程 superTrends）：
对因子信号沿时间维度评价其对未来收益的预测能力：Rank IC 与 IC 衰减、
预测回归（Newey-West HAC）、滚动 IC 与 ICIR、分层测试。

绘图（plotting）：个股日 K 线 + 因子曲线双子图，复用 src/utils.py 绘图函数。

用法::

    from factors import FactorLibrary, evaluate_factor, EvalConfig
    lib = FactorLibrary()
    s = lib.get("Alpha001", "BJ920010")
"""

from .eval_config import EvalConfig, nw_lag_auto, nw_lag_regression, nw_lag_rolling
from .evaluator import evaluate_factor, forward_returns
from .library import FactorLibrary
from .report import FactorReport

__all__ = [
    "FactorLibrary",
    "evaluate_factor",
    "EvalConfig",
    "FactorReport",
    "forward_returns",
    "nw_lag_auto",
    "nw_lag_regression",
    "nw_lag_rolling",
]
