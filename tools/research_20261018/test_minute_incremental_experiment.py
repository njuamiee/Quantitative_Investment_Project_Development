"""Tests of the timing boundary in the standalone minute incremental study."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minute_incremental_experiment as exp


class MinuteIncrementalTimingTest(unittest.TestCase):
    def test_future_and_immature_answers_cannot_enter_q4_training(self):
        original = pd.DataFrame({
            "date": pd.to_datetime(["2025-12-24", "2025-12-26", "2026-07-01"]),
            "stockid": ["BJ1", "BJ1", "BJ1"],
            "label_5": [0.03, 0.07, 0.11],
            "label_exit_5": pd.to_datetime(["2025-12-31", "2026-01-05", "2026-07-08"]),
        })
        with TemporaryDirectory() as folder:
            path = Path(folder) / "prices.parquet"
            original.to_parquet(path, index=False)
            with patch.object(exp, "PRICE_PATH", path):
                first = exp.mature_q4_labels()
                changed = original.copy()
                changed.loc[1:, "label_5"] = [900.0, -900.0]
                changed.to_parquet(path, index=False)
                second = exp.mature_q4_labels()
        self.assertEqual(len(first), 1)
        pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))

    def test_prediction_ignores_any_answer_columns_on_test_rows(self):
        n = 30
        frame = pd.DataFrame({
            "date": pd.Timestamp("2026-07-01"),
            "stockid": [f"BJ{i:06d}" for i in range(n)],
            "score_daily": np.linspace(-0.5, 0.5, n),
        })
        for j, feature in enumerate(exp.FEATURES):
            frame[feature] = np.sin(np.arange(n) * (j + 1) / 11)
        model = exp.Ridge(alpha=100).fit(frame[list(exp.FEATURES)], np.arange(n) / n)
        frame["label_5"] = 0.01
        first = exp.add_scores(frame, model, 0.25)
        frame["label_5"] = np.arange(n) * 1e8
        frame["label_exit_5"] = pd.Timestamp("2027-01-01")
        second = exp.add_scores(frame, model, 0.25)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
