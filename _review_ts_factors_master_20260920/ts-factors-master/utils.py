from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
import numpy as np
import pandas as pd


def simulate_futures(
    df: pd.DataFrame,
    signals: pd.Series | None = None,
    initial_capital: float = 1_000_000.0,
    margin_rate: float = 0.10, #保证金比率
    fee_rate: float = 0.00002, #万分之0.2
    contract_multiplier: float = 1000.0,  #每手多少股
    lots_per_trade: int | None = None, #可指定一个固定手数，不建议改
    capital_utilization: float = 0.5, #用于交易的总资产比例
    price_col: str = "close",
    signal_col: str = "Signal",
) -> pd.DataFrame:
    """
    模拟期货交易，输出各参数随时间变化的情况。

    信号约定（与 supertrend 输出的 Signal 列保持一致）：
    -  signal > 0 : 开多仓 / 平空仓转多仓，目标仓位 = +fixed_lots
    -  signal < 0 : 开空仓 / 平多仓转空仓，目标仓位 = -fixed_lots
    -  signal == 0 : 保持当前仓位不变

    交易规则：
    - 以指定价格列（默认 close）成交。
    - 开仓手数默认为固定手数。若 lots_per_trade 为 None，则根据
      当前总资产与 capital_utilization 自动计算，并在整个回测中保持不变。
    - 开仓时冻结保证金 = 成交价 × 手数 × 合约乘数 × 保证金率。
    - 开仓/平仓均收取手续费 = 成交名义价值 × 手续费率。
    - 平仓时释放保证金，已实现盈亏 = (平仓价 - 开仓价) × 仓位 × 合约乘数。
    - 持仓时记录未实现盈亏 = (当前价 - 开仓价) × 仓位 × 合约乘数。
    - 总资产 = 现金 + 冻结保证金 + 未实现盈亏。
    - 本函数不做日内追加保证金（margin call）处理，也不做部分成交处理。

    参数
    ----------
    df : pd.DataFrame
        输入 K 线数据，必须包含 price_col 指定的价格列。
    signals : pd.Series | None, 默认 None
        交易信号序列，长度与 df 一致。为 None 时读取 df[signal_col]。
    initial_capital : float, 默认 1_000_000.0
        初始资金。
    margin_rate : float, 默认 0.10
        保证金率（10%）。
    fee_rate : float, 默认 0.0001
        手续费率（万分之一）。
    contract_multiplier : float, 默认 1.0
        合约乘数（每点价值）。
    lots_per_trade : int | None, 默认 None
        每次开仓的手数。为 None 时，第一次开仓按 capital_utilization 自动计算。
    capital_utilization : float, 默认 0.5
        用于计算固定开仓手数的总资产使用比例，仅在 lots_per_trade 为 None 时生效。
        计算公式：fixed_lots = int((total_assets * capital_utilization) // margin_per_lot)。
    price_col : str, 默认 "close"
        成交价所在列名。
    signal_col : str, 默认 "Signal"
        当 signals 为 None 时，读取该列作为交易信号。

    返回
    -------
    pd.DataFrame
        在原 df 基础上追加以下列：
        - TargetPosition : 目标仓位
        - Position       : 实际仓位
        - TradeLots      : 本根 K 线成交手数（双向均计入）
        - TradePrice     : 本根 K 线成交价（未成交为 NaN）
        - TradeValue     : 本根 K 线成交名义价值
        - Fee            : 本根 K 线手续费
        - CumFee         : 累计手续费
        - RealizedPnl    : 本根 K 线已实现盈亏
        - CumRealizedPnl : 累计已实现盈亏
        - Cash           : 可用现金
        - Margin         : 冻结保证金
        - UnrealizedPnl  : 当前未实现盈亏
        - TotalAssets    : 总资产
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    if price_col not in df.columns:
        raise ValueError(f"df is missing price column: {price_col!r}")

    if signals is None:
        if signal_col not in df.columns:
            raise ValueError(
                f"signals is None and df is missing signal column: {signal_col!r}"
            )
        signals = df[signal_col]

    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas.Series")

    if len(signals) != len(df):
        raise ValueError("signals must have the same length as df")

    if not (0 < margin_rate <= 1):
        raise ValueError("margin_rate must be in (0, 1]")

    if fee_rate < 0:
        raise ValueError("fee_rate must be >= 0")

    if contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be > 0")

    if lots_per_trade is not None and lots_per_trade <= 0:
        raise ValueError("lots_per_trade must be > 0")

    if not (0 < capital_utilization <= 1):
        raise ValueError("capital_utilization must be in (0, 1]")

    prices = df[price_col].astype(float).to_numpy()
    sigs = signals.to_numpy()
    n = len(prices)

    target_pos = np.zeros(n, dtype=np.int64)
    positions = np.zeros(n, dtype=np.int64)
    trade_lots = np.zeros(n, dtype=np.int64)
    trade_prices = np.full(n, np.nan)
    trade_values = np.zeros(n, dtype=np.float64)
    fees = np.zeros(n, dtype=np.float64)
    cum_fees = np.zeros(n, dtype=np.float64)
    realized_pnls = np.zeros(n, dtype=np.float64)
    cum_realized_pnls = np.zeros(n, dtype=np.float64)
    cashs = np.zeros(n, dtype=np.float64)
    margins = np.zeros(n, dtype=np.float64)
    unrealized_pnls = np.zeros(n, dtype=np.float64)
    total_assets = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = np.nan
    cash = float(initial_capital)
    margin = 0.0
    cum_fee = 0.0
    cum_realized_pnl = 0.0
    fixed_lots = lots_per_trade

    # 第一次开仓且未指定固定手数时，按总资产使用比例计算固定手数
    if fixed_lots is None:
        price = prices[0]
        assert not np.isnan(price) and price > 1e-4
        margin_per_lot = price * contract_multiplier * margin_rate
        if margin_per_lot > 0:
            fixed_lots = max(1, int((cash + margin) * capital_utilization // margin_per_lot))
        else:
            fixed_lots = 1

    for i in range(n):
        price = prices[i]
        sig = sigs[i]

        # 确定目标仓位：信号为正/负时开相应方向，信号为 0 时保持原仓位
        if sig > 0:
            target = fixed_lots if fixed_lots is not None else 0
        elif sig < 0:
            target = -fixed_lots if fixed_lots is not None else 0
        else:
            target = position

        target_pos[i] = target

        if target != position and not np.isnan(price):
            # 先平掉旧仓位
            if position != 0:
                close_lots = abs(position)
                tv_close = close_lots * price * contract_multiplier
                fee_close = tv_close * fee_rate
                realized_pnl = (price - entry_price) * position * contract_multiplier

                cash = cash + margin + realized_pnl - fee_close
                cum_realized_pnl += realized_pnl
                margin = 0.0

                trade_lots[i] += close_lots
                trade_values[i] += tv_close
                fees[i] += fee_close
                cum_fee += fee_close
                realized_pnls[i] += realized_pnl

                position = 0
                entry_price = np.nan

            # 再开新仓位
            if target != 0:
                open_lots = abs(target)
                tv_open = open_lots * price * contract_multiplier
                fee_open = tv_open * fee_rate
                margin_req = tv_open * margin_rate

                cash = cash - margin_req - fee_open
                margin = margin_req
                entry_price = price
                position = target

                trade_lots[i] += open_lots
                trade_values[i] += tv_open
                fees[i] += fee_open
                cum_fee += fee_open

        # 计算当前未实现盈亏和总资产
        if position != 0 and not np.isnan(price) and not np.isnan(entry_price):
            unrealized_pnl = (price - entry_price) * position * contract_multiplier
        else:
            unrealized_pnl = 0.0

        positions[i] = position
        trade_prices[i] = price if trade_lots[i] != 0 else np.nan
        cum_fees[i] = cum_fee
        cum_realized_pnls[i] = cum_realized_pnl
        cashs[i] = cash
        margins[i] = margin
        unrealized_pnls[i] = unrealized_pnl
        total_assets[i] = cash + margin + unrealized_pnl

    result = df.copy()
    result["TargetPosition"] = target_pos
    result["Position"] = positions
    result["TradeLots"] = trade_lots
    result["TradePrice"] = trade_prices
    result["TradeValue"] = trade_values
    result["Fee"] = fees
    result["CumFee"] = cum_fees
    result["RealizedPnl"] = realized_pnls
    result["CumRealizedPnl"] = cum_realized_pnls
    result["Cash"] = cashs
    result["Margin"] = margins
    result["UnrealizedPnl"] = unrealized_pnls
    result["TotalAssets"] = total_assets

    return result


def evaluate_futures(
    sim_df: pd.DataFrame,
    initial_capital: float | None = None,
    risk_free_annual: float = 0.0,
    trading_days_per_year: float = 252.0,
) -> pd.Series:
    """
    根据 simulate_futures 输出的时间序列，计算期货交易绩效指标。

    指标
    ----------
    - AnnualReturn      : 年化收益（几何年化）
    - SharpeRatio       : 夏普率，基于每根 K 线总资产收益率
    - MaxDrawdown       : 最大回撤
    - CalmarRatio       : 卡玛比率 = 年化收益 / |最大回撤|
    - WinRate           : 胜率 = 盈利交易次数 / 总交易次数
    - ProfitLossRatio   : 盈亏比 = 平均盈利 / |平均亏损|
    - TotalTradingDays  : 总交易天数（按日期去重）
    - TotalTrades       : 总交易次数（ completed round trips ）
    - HoldingRatio      : 持仓时间占比 = 有仓位 K 线数 / 总 K 线数

    参数
    ----------
    sim_df : pd.DataFrame
        simulate_futures 的返回结果，必须包含 'TotalAssets'、'Position'、'RealizedPnl'。
    initial_capital : float | None, 默认 None
        初始资金。为 None 时取 sim_df['TotalAssets'].iloc[0]。
    risk_free_annual : float, 默认 0.0
        年化无风险利率，用于计算夏普率。
    trading_days_per_year : float, 默认 252.0
        一年交易日数，用于年化折算。

    返回
    -------
    pd.Series
        各项绩效指标。
    """
    required = {"TotalAssets", "Position", "RealizedPnl"}
    missing = required - set(sim_df.columns)
    if missing:
        raise ValueError(f"sim_df is missing required columns: {sorted(missing)}")

    if initial_capital is None:
        initial_capital = float(sim_df["TotalAssets"].iloc[0])

    total_assets = sim_df["TotalAssets"].astype(float)
    positions = sim_df["Position"].astype(int)
    n_bars = len(sim_df)

    # 总交易天数
    if isinstance(sim_df.index, pd.DatetimeIndex):
        trading_days = int(sim_df.index.normalize().nunique())
    else:
        trading_days = max(1, n_bars)

    # 收益率与年化
    final_assets = float(total_assets.iloc[-1])
    total_return = (final_assets - initial_capital) / initial_capital
    years = trading_days / trading_days_per_year
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = np.nan

    # 夏普率：基于每根 K 线的总资产收益率
    bar_returns = total_assets.pct_change().dropna()
    if len(bar_returns) > 1 and bar_returns.std() > 0:
        bars_per_day = n_bars / trading_days if trading_days > 0 else 1.0
        annual_factor = bars_per_day * trading_days_per_year
        rf_per_bar = risk_free_annual / annual_factor
        excess = bar_returns - rf_per_bar
        sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(annual_factor)
    else:
        sharpe = np.nan

    # 最大回撤
    running_peak = total_assets.cummax()
    drawdown = (total_assets - running_peak) / running_peak
    max_drawdown = float(drawdown.min())

    # 卡玛比率
    if max_drawdown != 0 and not np.isnan(annual_return):
        calmar = annual_return / abs(max_drawdown)
    else:
        calmar = np.nan

    # 已完成交易次数与盈亏统计
    prev_positions = positions.shift(1).fillna(0).astype(np.int64)
    close_mask = (prev_positions != 0) & (positions != prev_positions)
    trade_pnls = sim_df.loc[close_mask, "RealizedPnl"].astype(float)
    total_trades = int(close_mask.sum())
    winning_trades = int((trade_pnls > 0).sum())
    losing_trades = int((trade_pnls < 0).sum())

    if total_trades > 0:
        win_rate = winning_trades / total_trades
        avg_win = trade_pnls[trade_pnls > 0].mean()
        avg_loss = trade_pnls[trade_pnls < 0].mean()
        if pd.notna(avg_loss) and avg_loss != 0:
            profit_loss_ratio = abs(avg_win / avg_loss)
        else:
            profit_loss_ratio = np.nan
    else:
        win_rate = np.nan
        profit_loss_ratio = np.nan

    # 持仓时间占比
    holding_bars = int((positions != 0).sum())
    holding_ratio = holding_bars / n_bars if n_bars > 0 else np.nan

    metrics = {
        "AnnualReturn": annual_return,
        "SharpeRatio": sharpe,
        "MaxDrawdown": max_drawdown,
        "CalmarRatio": calmar,
        "WinRate": win_rate,
        "ProfitLossRatio": profit_loss_ratio,
        "TotalTradingDays": trading_days,
        "TotalTrades": total_trades,
        "HoldingRatio": holding_ratio,
    }
    return pd.Series(metrics)


# ---------------------------------------------------------------------------
# Resampling helpers: synthesize higher timeframe bars from 1-minute data
# ---------------------------------------------------------------------------
DEFAULT_SESSIONS = [["09:30", "11:30"], ["13:00", "15:00"]]


def _build_bar_intervals_for_day(
    trading_day: pd.Timestamp,
    minutes: int,
    sessions: list[list[str]],
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, bool]]:
    """
    Build [start, end) bar intervals for a single trading day.
    Returns tuples of (label, start, end, is_last_bar).

    当目标周期不小于当日交易总时长（默认 A 股时段共 240 分钟）时，
    整日合成为一根 K 线（日线），label 取当日首个时段的起点。
    """
    total_session_minutes = 0
    for start_time, end_time in sessions:
        s = pd.Timestamp(f"{trading_day.date()} {start_time}")
        e = pd.Timestamp(f"{trading_day.date()} {end_time}")
        total_session_minutes += int((e - s).total_seconds() // 60)
    if minutes >= total_session_minutes:
        day_start = pd.Timestamp(f"{trading_day.date()} {sessions[0][0]}")
        day_end = pd.Timestamp(f"{trading_day.date()} {sessions[-1][1]}")
        return [(day_start, day_start, day_end, True)]

    step_rule = f"{int(minutes)}min"
    labels: list[pd.Timestamp] = []
    for start_time, end_time in sessions:
        session_start = pd.Timestamp(f"{trading_day.date()} {start_time}")
        session_end = pd.Timestamp(f"{trading_day.date()} {end_time}")
        session_labels = pd.date_range(
            start=session_start, end=session_end, freq=step_rule, inclusive="left"
        )
        labels.extend(session_labels.to_list())
    if not labels:
        return []
    labels = sorted(set(labels))
    last_session_end = pd.Timestamp(f"{trading_day.date()} {sessions[-1][1]}")
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, bool]] = []
    for idx, label in enumerate(labels):
        if idx < len(labels) - 1:
            end = labels[idx + 1]
            intervals.append((label, label, end, False))
        else:
            intervals.append((label, label, last_session_end, True))
    return intervals


def resample_minute_bars(
    df: pd.DataFrame,
    minutes: int,
    time_col: str = "time",
    sessions: list[list[str]] | None = None,
) -> pd.DataFrame:
    """
    从 1 分钟 K 线合成更大时间尺度的 K 线（如 10min / 20min / 30min）。
    当 minutes 不小于当日交易总时长（默认 240 分钟）时，整日合成为一根日 K 线。

    参数
    ----------
    df : pd.DataFrame
        输入的分钟级 K 线数据，必须包含 open/high/low/close/volume 列。
        索引可以是 DatetimeIndex，也可以由 time_col 列指定时间。
    minutes : int
        目标周期分钟数，例如 10 表示合成 10 分钟线。
    time_col : str, 默认 "time"
        当索引不是 DatetimeIndex 时，使用此列作为时间列。
    sessions : list[list[str]] | None, 默认 None
        交易时段，例如 [["09:30", "11:30"], ["13:00", "15:00"]]。
        为 None 时使用默认 A 股交易时段。

    返回
    -------
    pd.DataFrame
        合成后的 OHLCV 数据，索引为 DatetimeIndex，按时间升序排列。
        只聚合有效行情行，不保留 remark/data_gap 等标记行。
    """
    if minutes <= 1:
        return df.copy()
    if sessions is None:
        sessions = DEFAULT_SESSIONS

    # Normalize a copy so we don't mutate the caller's data.
    frame = df.copy()

    # Build DatetimeIndex from the time column if necessary.
    if not isinstance(frame.index, pd.DatetimeIndex):
        if time_col not in frame.columns:
            raise ValueError(
                f"Index is not a DatetimeIndex and column {time_col!r} is missing."
            )
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[time_col]).set_index(time_col)

    if frame.index.empty:
        return frame.iloc[0:0].copy()

    # Normalize column names to lower case for consistent aggregation.
    lower_cols = {c.lower(): c for c in frame.columns}
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(lower_cols)
    if missing:
        raise ValueError(f"Input frame is missing required columns: {sorted(missing)}")

    rename_map = {lower_cols[c]: c for c in required}
    frame = frame.rename(columns=rename_map)

    # Strip any non-trading marker rows (e.g. data_gap) by keeping rows with
    # a valid numeric close only. This intentionally ignores data_gap logic.
    frame = frame[frame.index.notna()]
    frame = frame[pd.to_numeric(frame["close"], errors="coerce").notna()]

    if frame.empty:
        return frame.iloc[0:0].copy()

    output_rows: list[dict] = []
    output_index: list[pd.Timestamp] = []

    for trading_day, day_frame in frame.groupby(frame.index.normalize()):
        intervals = _build_bar_intervals_for_day(pd.Timestamp(trading_day), minutes, sessions)
        for label, start, end, include_right_end in intervals:
            if include_right_end:
                window = day_frame.loc[(day_frame.index >= start) & (day_frame.index <= end)]
            else:
                window = day_frame.loc[(day_frame.index >= start) & (day_frame.index < end)]

            if window.empty:
                continue

            aggregated = {
                "open": float(window["open"].iloc[0]),
                "high": float(window["high"].max()),
                "low": float(window["low"].min()),
                "close": float(window["close"].iloc[-1]),
                "volume": float(window["volume"].sum()),
            }
            output_rows.append(aggregated)
            output_index.append(label)

    if not output_rows:
        return frame.iloc[0:0].copy()

    result = pd.DataFrame(output_rows, index=pd.DatetimeIndex(output_index)).sort_index()
    return result


def forward_adjust_daily_bars(
    df: pd.DataFrame,
    actions: pd.DataFrame,
    time_col: str = "time",
    price_cols: tuple[str, ...] = ("pre_close", "open", "high", "low", "close"),
) -> pd.DataFrame:
    """
    按分红/拆分记录对 K 线做前复权（锚定最新价，历史价格按比例调整）。

    复权因子口径：对除权除息日 d 的公司行为（现金红利 D 元/份，拆分比例 s，
    即 1 份变为 s 份，s>1 为拆分、s<1 为合并），当日因子为

        f_d = (C - D) / (C * s)

    其中 C 为 d 之前最后一个交易日的未复权收盘价。任一 bar 所在日 t 的累计因子为
    所有除息日晚于 t 的行为因子之积：

        F(t) = ∏_{d > t} f_d

    前复权价 = 未复权价 × F(t)。最近一次行为之后的 bar 因子为 1（价格与原始
    数据一致），历史价格相应缩放，使收益率序列在除息日连续。本函数为日频设计，
    对按日期对齐的其它周期同样适用（逐 bar 按其所在日取因子）。

    参数
    ----------
    df : pd.DataFrame
        原始（未复权）K 线数据。索引为 DatetimeIndex，或包含 time_col 指定的
        日期列；价格列名大小写不敏感（OPEN/Open/open 均可）。行不必有序，
        返回值按日期升序排列。
    actions : pd.DataFrame
        复权信息，列名大小写不敏感：
        - date     : 除权除息日（必需）；
        - dividend : 每份现金红利（元），缺省按 0 处理；
        - split    : 拆分比例（1 份变为 split 份），缺省或取 0/1 表示无拆分。
    time_col : str, 默认 "time"
        df 索引非 DatetimeIndex 时使用的日期列名。
    price_cols : tuple[str, ...], 默认 ("pre_close", "open", "high", "low", "close")
        参与复权的价格列（按小写名匹配，不存在的列自动跳过）。pre_close 在
        除息日通常为交易所公布的除权参考价（= (前收盘-红利)/拆分比例），与其它
        价格列乘同一因子后口径保持一致；volume 等非价格列保持原值。

    返回
    -------
    pd.DataFrame
        前复权后的数据，列名与列顺序不变。
    """
    frame = df.copy()

    # 日期序列（与行位置对齐），统一按日期升序
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = frame.index.normalize()
    else:
        if time_col not in frame.columns:
            raise ValueError(f"索引非 DatetimeIndex 且缺少日期列: {time_col!r}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame[time_col], errors="coerce")).normalize()
    if dates.hasnans:
        raise ValueError("存在无法解析的日期")
    order = dates.argsort(kind="stable")
    if not (order == np.arange(len(frame))).all():
        frame = frame.iloc[order]
        dates = dates[order]

    # 价格列（大小写不敏感）；close 为计算分红因子所必需
    lower_cols = {c.lower(): c for c in frame.columns}
    if "close" not in lower_cols:
        raise ValueError("缺少 close 列，无法计算分红因子")
    target_cols = [lower_cols[c] for c in price_cols if c in lower_cols]
    close_arr = pd.to_numeric(
        frame[lower_cols["close"]], errors="coerce"
    ).to_numpy(dtype=float)

    # 解析公司行为：date 必需，dividend/split 缺省按 0/1
    lower_act = {str(c).lower(): c for c in actions.columns}
    if "date" not in lower_act:
        raise ValueError("actions 缺少 date 列")
    ex_dates = pd.DatetimeIndex(
        pd.to_datetime(actions[lower_act["date"]], errors="coerce")
    ).normalize()
    if ex_dates.hasnans:
        raise ValueError("actions 存在无法解析的除权除息日")
    if "dividend" in lower_act:
        dividends = pd.to_numeric(actions[lower_act["dividend"]], errors="coerce").fillna(0.0)
    else:
        dividends = pd.Series(0.0, index=actions.index)
    if "split" in lower_act:
        splits = pd.to_numeric(actions[lower_act["split"]], errors="coerce").fillna(1.0)
    else:
        splits = pd.Series(1.0, index=actions.index)

    # 逐行为累计因子：F(t) = ∏_{d > t} f_d
    factor = np.ones(len(frame), dtype=float)
    for d_ex, div, spl in zip(ex_dates, dividends.to_numpy(), splits.to_numpy()):
        if div < 0:
            raise ValueError(f"现金红利不能为负: {d_ex.date()} dividend={div}")
        if spl < 0:
            raise ValueError(f"拆分比例不能为负: {d_ex.date()} split={spl}")
        has_div = div != 0
        has_spl = spl not in (0, 1)
        if not has_div and not has_spl:
            continue
        mask = dates < d_ex
        if not mask.any():
            continue  # 行为早于全部数据，对结果无影响
        f = 1.0
        if has_div:
            hist = close_arr[mask]
            hist = hist[~np.isnan(hist)]
            if len(hist) == 0 or hist[-1] <= 0:
                raise ValueError(f"{d_ex.date()} 之前无有效收盘价，无法计算分红因子")
            prev_close = hist[-1]
            if prev_close - div <= 0:
                raise ValueError(
                    f"{d_ex.date()} 现金红利 {div} 不小于前收盘价 {prev_close}，数据异常"
                )
            f *= (prev_close - div) / prev_close
        if has_spl:
            f *= 1.0 / spl
        factor[mask] *= f

    for col in target_cols:
        frame[col] = (
            pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float) * factor
        )
    return frame


def plot_candles_with_ema(
    data: pd.DataFrame,
    ema_cols: list[str] | None = None,
    title: str = "Candlestick with EMAs",
    ax: plt.Axes | None = None,
    extra_lines: dict[str, pd.Series] | None = None,
    extra_line_colors: dict[str, str] | None = None,
    signal_col: str | None = None,
    long_marker: str = "^",
    short_marker: str = "v",
    long_color: str = "red",
    short_color: str = "green",
) -> plt.Axes:
    """
    Draw a candlestick chart with optional EMA lines, extra overlay lines and
    trading-signal scatter points.

    Parameters
    ----------
    data : pd.DataFrame
        Data with DatetimeIndex and OHLC columns (case-insensitive).
    ema_cols : list[str] | None
        EMA columns to draw on top of candles.
    title : str
        Chart title (applied to the provided or created Axes).
    ax : plt.Axes | None
        Axes to draw on. If None, a new standalone figure is created.
    extra_lines : dict[str, pd.Series] | None
        Additional lines to overlay, e.g. {"SuperTrend": data["SuperTrend"]}}.
    extra_line_colors : dict[str, str] | None
        Color for each extra line. Missing entries use the default palette.
    signal_col : str | None
        Column with signal values: 1 for long, -1 for short, others ignored.
    long_marker, short_marker : str
        Marker style for long/short signals.
    long_color, short_color : str
        Marker color for long/short signals.

    Returns
    -------
    plt.Axes
        The Axes that was drawn on.
    """
    if data.empty:
        raise ValueError("data is empty")

    lower_cols = {c.lower(): c for c in data.columns}
    required = {"open", "high", "low", "close"}
    missing = required - set(lower_cols)
    if missing:
        raise ValueError(f"data is missing required OHLC columns: {sorted(missing)}")

    ema_cols = ema_cols or []
    for c in ema_cols:
        if c not in data.columns:
            raise ValueError(f"EMA column not found: {c}")

    extra_lines = extra_lines or {}
    for name, series in extra_lines.items():
        if not isinstance(series, pd.Series):
            raise ValueError(f"extra_lines['{name}'] must be a pandas.Series")

    if signal_col is not None and signal_col not in data.columns:
        raise ValueError(f"signal column not found: {signal_col}")

    frame = data.copy()
    open_col = lower_cols["open"]
    high_col = lower_cols["high"]
    low_col = lower_cols["low"]
    close_col = lower_cols["close"]

    x = np.arange(len(frame), dtype=float)
    width = 0.6
    up = frame[close_col] >= frame[open_col]
    down = ~up

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(18, 8))

    ax.vlines(
        x,
        frame[low_col].to_numpy(),
        frame[high_col].to_numpy(),
        color="black",
        linewidth=0.5,
        alpha=0.6,
        zorder=1,
    )

    ax.bar(
        x[up.to_numpy()],
        (frame.loc[up, close_col] - frame.loc[up, open_col]).to_numpy(),
        bottom=frame.loc[up, open_col].to_numpy(),
        width=width,
        color="#d62728",
        edgecolor="#d62728",
        alpha=0.9,
        zorder=2,
    )
    ax.bar(
        x[down.to_numpy()],
        (frame.loc[down, close_col] - frame.loc[down, open_col]).to_numpy(),
        bottom=frame.loc[down, open_col].to_numpy(),
        width=width,
        color="#2ca02c",
        edgecolor="#2ca02c",
        alpha=0.9,
        zorder=2,
    )

    colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
    for idx, ema_col in enumerate(ema_cols):
        ax.plot(
            x,
            frame[ema_col].to_numpy(),
            label=ema_col,
            linewidth=1.5,
            color=colors[idx % len(colors)],
            zorder=3,
        )

    extra_line_colors = extra_line_colors or {}
    for idx, (name, series) in enumerate(extra_lines.items()):
        color = extra_line_colors.get(name, colors[(idx + len(ema_cols)) % len(colors)])
        ax.plot(
            x,
            series.to_numpy(),
            label=name,
            linewidth=1.5 if name != "SuperTrend" else 1.8,
            color=color,
            alpha=0.9,
            zorder=3,
        )

    if signal_col is not None:
        long_sig = frame[signal_col] == 1
        short_sig = frame[signal_col] == -1
        if long_sig.any():
            ax.scatter(
                x[long_sig.to_numpy()],
                frame.loc[long_sig, close_col].to_numpy()*1.01,
                marker=long_marker,
                color=long_color,
                s=70,
                label="Long Signal",
                zorder=4,
            )
        if short_sig.any():
            ax.scatter(
                x[short_sig.to_numpy()],
                frame.loc[short_sig, close_col].to_numpy()*0.99,
                marker=short_marker,
                color=short_color,
                s=70,
                label="Short Signal",
                zorder=4,
            )

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    # x 轴使用行号，刻度标签通过 FuncFormatter 格式化为时间
    _apply_time_xticks(ax, frame.index)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    # 简单自测：构造一个先涨后跌的序列，配合 SuperTrend 信号跑模拟
    from tmp_code.signals.supertrend import supertrend

    test_df = pd.DataFrame(
        {
            "open": [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
            "high": [102, 103, 104, 103, 102, 101, 100, 101, 102, 103],
            "low": [99, 100, 101, 100, 99, 98, 97, 98, 99, 100],
            "close": [101, 102, 103, 100, 99, 98, 97, 99, 100, 102],
        },
        index=pd.date_range("2026-01-01 09:30", periods=10, freq="1min"),
    )

    st = supertrend(test_df, atr_period=3, multiplier=1.5, use_wilder=True)
    df_signal = pd.concat([test_df, st[["Signal"]]], axis=1)

    sim = simulate_futures(df_signal, initial_capital=1_000_000.0)
    print(
        sim[
            [
                "close",
                "Signal",
                "Position",
                "TradeLots",
                "Fee",
                "Cash",
                "Margin",
                "UnrealizedPnl",
                "TotalAssets",
            ]
        ]
    )
    print(f"\n最终总资产: {sim['TotalAssets'].iloc[-1]:,.2f}")
    print(f"累计手续费: {sim['CumFee'].iloc[-1]:,.2f}")

    metrics = evaluate_futures(sim)
    print("\n绩效指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 简单自测：将 1 分钟数据重采样为 2 分钟线
    print("\n--- resample_minute_bars 自测 ---")
    one_min = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [101, 102, 103, 104],
            "volume": [10, 20, 30, 40],
        },
        index=pd.date_range("2026-01-01 09:30", periods=4, freq="1min"),
    )
    two_min = resample_minute_bars(one_min, minutes=2)
    print(two_min)
    assert len(two_min) == 2, "Expected 2 bars"
    assert two_min["open"].iloc[0] == 100.0
    assert two_min["close"].iloc[-1] == 104.0
    assert two_min["volume"].sum() == 100.0
    print("resample_minute_bars 自测通过")


# ============================================================================
# 多尺度量化系统新增绘图接口
# ============================================================================


def _ensure_chinese_font() -> None:
    """为 Matplotlib 选择可用中文字体，并修复负号显示。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _apply_time_xticks(ax: plt.Axes, index: pd.Index, tick_count: int = 12) -> None:
    """x 轴按行号绘制时，用 FuncFormatter 将行号刻度格式化为时间标签。

    DatetimeIndex 显示 ``yy-MM-dd HH:mm``（截取自 str(Timestamp) 的 [2:16]）；
    非时间索引退化为显示行号本身。
    """
    n = len(index)
    if n <= 0:
        return
    step = max(1, n // tick_count)
    ticks = np.arange(0, n, step, dtype=int)
    if ticks[-1] != n - 1:
        ticks = np.append(ticks, n - 1)

    ntot = n
    if isinstance(index, pd.DatetimeIndex):
        def format_time(x_pos, pos=None):
            return "" if x_pos < 0 or x_pos >= ntot else str(index[int(x_pos)])[2:16]
    else:
        def format_time(x_pos, pos=None):
            return "" if x_pos < 0 or x_pos >= ntot else str(int(x_pos))

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))


def add_right_click_vertical_lines(
    fig: plt.Figure,
    axes: list[plt.Axes] | None = None,
    color: str = "#666666",
    linestyle: str = "--",
    linewidth: float = 1.0,
    alpha: float = 0.75,
) -> plt.Figure:
    """为图表添加右键点击竖线交互。

    每次右键点击都会在当前 fig 内所有子图上追加一条竖线，便于对齐多个子图的
    时间点/索引位置。这个函数可复用于回测图、因子图等多子图布局。
    """
    #print('add_right_click_vertical_lines: 右键点击图表可添加竖线，便于对齐多个子图的时间点/索引位置。\n', axes)
    if axes is None:
        axes = list(fig.axes)

    if not axes:
        raise ValueError("fig has no axes to attach right-click vertical guide lines")

    def _handler(event):
        if event.button != 3 or event.inaxes is None:
            return
        x = event.xdata
        if x is None:
            return

        for ax in axes:
            if not ax.lines:
                ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=99)
            else:
                # 保持同一点击位置的线在最上层，避免被遮住。
                ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=99)

    fig._right_click_vline_handler = _handler
    fig.canvas.mpl_connect("button_press_event", _handler)
    return fig


def plot_candles(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str = "K Line",
    ohlc_cols: tuple[str, str, str, str] = ("target_open", "target_high", "target_low", "target_close"),
    ema_cols: list[str] | None = None,
) -> plt.Axes:
    """画 K 线图（可叠加 EMA 均线）。

    复用 ``plot_candles_with_ema``，仅提供 OHLC 列名映射；
    ``ema_cols`` 中的列需已存在于 df，将原样叠加绘制。
    """
    open_col, high_col, low_col, close_col = ohlc_cols
    required = {open_col, high_col, low_col, close_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少 OHLC 列: {sorted(missing)}")

    ema_cols = ema_cols or []
    missing_ema = [c for c in ema_cols if c not in df.columns]
    if missing_ema:
        raise ValueError(f"缺少 EMA 列: {missing_ema}")

    sub = df[list(ohlc_cols) + list(ema_cols)].copy()
    sub.columns = ["open", "high", "low", "close"] + list(ema_cols)
    return plot_candles_with_ema(sub, title=title, ax=ax, ema_cols=ema_cols)


def plot_signals(
    df: pd.DataFrame,
    signal_col: str = "signal_fused",
    ax: plt.Axes | None = None,
    markersize: int = 8,
    threshold: float = 0.6,
) -> plt.Axes:
    """在 K 线图上绘制买入/卖出信号点。

    信号 > threshold 标记为买入（红色上三角），
    信号 < -threshold 标记为卖出（绿色下三角）。
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(18, 4))

    x = np.arange(len(df))
    close = df["target_close"]

    long_mask = df[signal_col] > threshold
    short_mask = df[signal_col] < -threshold

    ax.scatter(
        x[long_mask],
        close[long_mask],
        marker="^",
        color="red",
        s=markersize ** 2,
        label="Long signal",
        zorder=5,
    )
    ax.scatter(
        x[short_mask],
        close[short_mask],
        marker="v",
        color="green",
        s=markersize ** 2,
        label="Short signal",
        zorder=5,
    )
    ax.legend()
    if standalone:
        ax.set_title(f"Signals ({signal_col})")
    return ax


def plot_equity_curve(
    df: pd.DataFrame,
    equity_col: str = "equity",
    ax: plt.Axes | None = None,
    title: str = "Equity Curve",
    benchmark_col: str | None = None,
) -> plt.Axes:
    """绘制总资产随时间变化曲线（x 轴使用行号，刻度显示时间）。

    ``benchmark_col`` 非空时叠加该列作为基准曲线（如买入持有）。
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(18, 4))

    x = np.arange(len(df))
    ax.plot(x, df[equity_col].to_numpy(), label="Equity", linewidth=1.2)
    if benchmark_col is not None:
        if benchmark_col not in df.columns:
            raise ValueError(f"列不存在: {benchmark_col}")
        ax.plot(
            x, df[benchmark_col].to_numpy(), label=benchmark_col, linewidth=1.0, alpha=0.8
        )
    ax.axhline(
        df[equity_col].iloc[0], color="gray", linestyle="--", linewidth=0.8, label="Initial"
    )
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", alpha=0.25)
    _apply_time_xticks(ax, df.index)
    return ax


def add_line_to_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    label: str | None = None,
    **plot_kwargs,
) -> plt.Axes:
    """通用接口：在主图 ax 上绘制任意列曲线（x 轴使用行号）。

    方便各模块（如信号发生器、融合器）绘制自己的曲线。
    """
    if col not in df.columns:
        raise ValueError(f"列不存在: {col}")
    label = label or col
    x = np.arange(len(df))
    ax.plot(x, df[col].to_numpy(), label=label, **plot_kwargs)
    ax.legend()
    return ax


def plot_signal_curve(
    df: pd.DataFrame,
    ax: plt.Axes,
    signal_col: str = "signal_fused",
) -> plt.Axes:
    """在指定坐标轴上绘制信号曲线（x 轴使用行号，刻度显示时间）。"""
    if signal_col not in df.columns:
        raise ValueError(f"列不存在: {signal_col}")
    x = np.arange(len(df))
    ax.plot(x, df[signal_col].to_numpy(), label="Signal", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Signal")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.25)
    _apply_time_xticks(ax, df.index)
    return ax


def plot_position_curve(
    df: pd.DataFrame,
    ax: plt.Axes,
    position_col: str = "position",
) -> plt.Axes:
    """在指定坐标轴上绘制持仓曲线（x 轴使用行号，刻度显示时间）。"""
    if position_col not in df.columns:
        raise ValueError(f"列不存在: {position_col}")
    x = np.arange(len(df))
    ax.plot(x, df[position_col].to_numpy(), label="Position", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Position")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.25)
    _apply_time_xticks(ax, df.index)
    return ax


def plot_backtest_result(
    result_df: pd.DataFrame,
    signal_col: str = "signal_fused",
    equity_col: str = "equity",
    position_col: str = "shares",
    benchmark_col: str | None = None,
    ema_periods: tuple = (5, 20),
    boll: tuple[int, float] | None = None,
    title: str = "Backtest Result",
    figsize: tuple = (18, 10),
    save_path: str | None = None,
    show: bool = False,
) -> str | None:
    """一站式绘制回测结果（x 轴统一为行号，刻度显示时间）。

    从上到下四个子图：
      1. K 线 + EMA 均线（按 ``ema_periods`` 在 target_close 上计算）；
         ``boll`` 非空时再叠加布林线（period, n_std）：mid=rolling(period).mean()，
         upper/lower = mid ± n_std·rolling(period).std()，与因子库 Bolling 口径一致
      2. 融合信号曲线
      3. 持仓 position
      4. 总资产 equity 曲线（``benchmark_col`` 非空时叠加基准曲线，如买入持有）

    ``save_path`` 非空时不弹窗，改为存盘（自动创建父目录、关闭 figure）并返回
    该路径；为 None 时维持 ``plt.show()``，返回 None。``show`` 为 True 时即使
    ``save_path`` 非空也在存盘后 ``plt.show()`` 阻塞显示，直到用户关闭图形。
    """
    df = result_df.copy()
    _ensure_chinese_font()
    ema_cols = []
    for p in ema_periods:
        col = f"EMA{int(p)}"
        df[col] = df["target_close"].ewm(span=int(p), adjust=False).mean()
        ema_cols.append(col)
    if boll is not None:
        boll_period, boll_n_std = int(boll[0]), float(boll[1])
        boll_mid = df["target_close"].rolling(boll_period).mean()
        boll_std = df["target_close"].rolling(boll_period).std()
        df["BOLL_mid"] = boll_mid
        df["BOLL_upper"] = boll_mid + boll_n_std * boll_std
        df["BOLL_lower"] = boll_mid - boll_n_std * boll_std
        ema_cols += ["BOLL_mid", "BOLL_upper", "BOLL_lower"]

    fig, axes = plt.subplots(
        4, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1]}
    )
    plot_candles(df, ax=axes[0], title="Price & EMA", ema_cols=ema_cols)
    plot_signal_curve(df, ax=axes[1], signal_col=signal_col)
    plot_position_curve(df, ax=axes[2], position_col=position_col)
    plot_equity_curve(df, equity_col=equity_col, ax=axes[3], title="Equity",
                      benchmark_col=benchmark_col)

    add_right_click_vertical_lines(fig, list(axes))
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)
        return save_path
    plt.show()
    return None



def plot_daily_returns_bar(
    daily_df: pd.DataFrame,
    return_col: str = "total_return",
    title: str = "Daily Test Returns",
    figsize: tuple = (18, 5),
    save_path: str | None = None,
) -> str | None:
    """绘制逐 test 日收益柱状图和累计总资产曲线。

    正值红色、负值绿色（A 股习惯）。``save_path`` 非空时不弹窗，改为存盘
    （自动创建父目录、关闭 figure）并返回该路径；为 None 时维持 ``plt.show()``。
    """
    if return_col not in daily_df.columns:
        raise ValueError(f"列不存在: {return_col}")
    returns = pd.to_numeric(daily_df[return_col], errors="coerce")
    if returns.isna().any():
        raise ValueError(f"列包含非数值或缺失收益: {return_col}")

    # 每日收益按复利累积，净值从 1.0 开始。
    equity = (1.0 + returns).cumprod()
    n_days = len(returns)
    total_return = equity.iloc[-1] - 1.0 if n_days else 0.0
    annual_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0 if n_days else 0.0
    std = returns.std()
    sharpe = np.sqrt(252.0) * returns.mean() / std if std > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = drawdown.min() if n_days else 0.0
    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else np.inf

    print(
        f"[{title}] 总收益: {total_return * 100:.2f}%, "
        f"年化收益: {annual_return * 100:.2f}%, "
        f"最大回撤: {max_drawdown * 100:.2f}%, "
        f"夏普率: {sharpe:.3f}, 卡玛系数: {calmar:.3f}"
    )

    _ensure_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], max(figsize[1], 8)), sharex=True)
    values = returns.to_numpy() * 100.0
    colors = np.where(values >= 0, "red", "green")
    axes[0].bar(np.arange(len(daily_df)), values, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(title)
    axes[0].set_ylabel("daily return (%)")
    axes[0].grid(True, linestyle="--", alpha=0.25)

    axes[1].plot(np.arange(len(daily_df)), equity.to_numpy(), color="tab:blue", linewidth=1.5)
    axes[1].set_title("Cumulative Equity")
    axes[1].set_ylabel("equity (start=1.0)")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    _apply_time_xticks(axes[1], daily_df.index)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
    plt.show()
    return None
