"""Tests for historical label maturity, paired-missing IC and train-only PCA."""
import unittest

import numpy as np
import pandas as pd

from models import Candidate, daily_factor_ic, fit_candidate, matured, pca_transform, screen


class ModelBoundaryTests(unittest.TestCase):
    def test_future_or_unmatured_answers_cannot_change_fit(self):
        dates = pd.date_range("2025-01-01", periods=16, freq="B")
        rows = []
        rng = np.random.default_rng(10)
        for di, date in enumerate(dates):
            for stock in range(8):
                rows.append({"date": date, "stockid": f"S{stock}",
                             "label_exit_5": dates[min(di + 2, len(dates) - 1)],
                             "label_5": stock / 10 + rng.normal(0, .01),
                             "x_a": stock / 10 + rng.normal(0, .01),
                             "x_b": (7 - stock) / 10 + rng.normal(0, .01),
                             "x_c": rng.normal(0, .2)})
        panel = pd.DataFrame(rows)
        cutoff = dates[11]
        altered = panel.copy()
        altered.loc[altered.label_exit_5 >= cutoff, "label_5"] = 1e6
        history = dates[:11]
        a = matured(panel, history, cutoff)
        b = matured(altered, history, cutoff)
        pd.testing.assert_frame_equal(a, b)
        features = ["x_a", "x_b", "x_c"]
        cols_a, card_a = screen(a, features, None, maximum=2, deduplicate=False)
        cols_b, card_b = screen(b, features, None, maximum=2, deduplicate=False)
        self.assertEqual(cols_a, cols_b)
        candidate = Candidate("unit", "pca_ridge", 10, 2, None)
        apply = panel[panel.date == dates[11]].copy()
        pred_a, _, _ = fit_candidate(candidate, a, apply, cols_a, card_a)
        pred_b, _, _ = fit_candidate(candidate, b, apply, cols_b, card_b)
        np.testing.assert_allclose(pred_a, pred_b, rtol=0, atol=1e-12)

    def test_daily_ic_uses_only_complete_pairs(self):
        date = pd.Timestamp("2025-01-01")
        x = [1., 2., np.nan, 4., 5., 6.]
        y = [1., 2., -100., 4., 5., 6.]
        frame = pd.DataFrame({"date": [date] * 6, "target_now": y, "x_a": x})
        _, ic = daily_factor_ic(frame, ["x_a"])
        self.assertAlmostEqual(float(ic[0, 0]), 1.0, places=6)

    def test_pca_fits_train_only_and_keeps_correlated_inputs(self):
        rng = np.random.default_rng(7)
        train = rng.normal(size=(30, 5)).astype(np.float32)
        apply = rng.normal(loc=100, size=(10, 5)).astype(np.float32)
        a, z, pca, _ = pca_transform(train, apply)
        np.testing.assert_allclose(pca.mean_, train.mean(axis=0), atol=1e-6)
        a2, _, pca2, _ = pca_transform(train, apply + 1000)
        np.testing.assert_allclose(a, a2, atol=1e-6)
        np.testing.assert_allclose(pca.components_, pca2.components_, atol=1e-6)
        self.assertEqual(z.shape[0], len(apply))
        # The PCA branch deliberately retains two almost identical columns.
        records = []
        for di, date in enumerate(pd.date_range("2025-01-01", periods=8, freq="B")):
            for stock in range(8):
                records.append({"date": date, "target_now": stock / 8,
                                "x_a": stock / 8 + di / 10000,
                                "x_b": stock / 8 + di / 9000})
        frame = pd.DataFrame(records)
        raw, _ = screen(frame, ["x_a", "x_b"], None, maximum=2, deduplicate=True)
        pca_input, _ = screen(frame, ["x_a", "x_b"], None, maximum=2, deduplicate=False)
        self.assertEqual(len(raw), 1)
        self.assertEqual(len(pca_input), 2)


if __name__ == "__main__":
    unittest.main()
