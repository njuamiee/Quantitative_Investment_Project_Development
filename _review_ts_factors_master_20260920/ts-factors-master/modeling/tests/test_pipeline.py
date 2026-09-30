from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.train_lightgbm_hmm import load_prices, moving_block_mean_ci


def _write_price(path: Path, closes: np.ndarray) -> None:
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    frame = pd.DataFrame(
        {
            "日期": dates.strftime("%Y-%m-%d"),
            "股票代码": 920001,
            "开盘": closes - 0.5,
            "收盘": closes,
            "最高": closes + 1.0,
            "最低": closes - 1.0,
            "成交量": 100_000.0,
            "成交额": closes * 100_000.0,
            "换手率": 2.0,
        }
    )
    frame.to_csv(path, index=False, encoding="gb18030")


class PipelineTests(unittest.TestCase):
    def test_label_uses_next_open_and_fifth_market_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "920001-test.csv"
            closes = np.arange(10.0, 40.0)
            _write_price(path, closes)
            prices, _ = load_prices(Path(directory))
            first = prices.iloc[0]
            self.assertEqual(first["entry_date"], prices.iloc[1]["date"])
            self.assertEqual(first["exit_date"], prices.iloc[5]["date"])
            expected = closes[5] / (closes[1] - 0.5) - 1.0
            self.assertAlmostEqual(first["raw_return_5d"], expected)

    def test_future_price_changes_do_not_change_historical_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "920001-test.csv"
            original = np.arange(10.0, 50.0)
            _write_price(path, original)
            before, _ = load_prices(Path(directory))
            changed = original.copy()
            changed[30:] = changed[30:] * 10
            _write_price(path, changed)
            after, _ = load_prices(Path(directory))
            columns = [
                "ret1",
                "mom5",
                "mom10",
                "mom20",
                "vol10",
                "vol20",
                "amount_ma20",
                "range20",
                "log_illiq20",
            ]
            pd.testing.assert_frame_equal(
                before.loc[:29, columns], after.loc[:29, columns]
            )

    def test_block_bootstrap_is_deterministic(self):
        values = np.linspace(0.4, 0.6, 50)
        self.assertEqual(moving_block_mean_ci(values), moving_block_mean_ci(values))


if __name__ == "__main__":
    unittest.main()
