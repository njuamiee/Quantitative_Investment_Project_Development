"""Synthetic tests for the offline proposal/selection/independent-test boundary."""
import json
import unittest

import numpy as np
import pandas as pd

from auto_factor_lab import (Protocol, allowed_inputs, build_local_prompt,
                             parse_proposals, proposal_from_local_llm, run_lab,
                             split_mature)


def fixture() -> tuple[pd.DataFrame, Protocol, list[dict]]:
    rng = np.random.default_rng(1018)
    dates = pd.bdate_range("2025-01-02", periods=100)
    records = []
    for i, date in enumerate(dates):
        exit_date = dates[i + 5] if i + 5 < len(dates) else pd.NaT
        for stock in range(24):
            a, b = rng.normal(size=2)
            records.append({"date": date, "stockid": f"BJ{stock:06d}",
                            "x_a": a, "x_b": b, "x_noise": rng.normal(),
                            "label_5": a * b + rng.normal(scale=.12),
                            "label_exit_5": exit_date})
    protocol = Protocol(str(dates[0].date()), str(dates[40].date()),
                        str(dates[65].date()), str(dates[95].date()),
                        min_days=15)
    proposals = [{"id": "interaction_ab", "expression": "x_a * x_b",
                  "hypothesis": "An interaction rather than either input alone",
                  "proposed_by": "synthetic-test"}]
    return pd.DataFrame(records), protocol, proposals


class AutoFactorLabTests(unittest.TestCase):
    def test_feedback_mode_never_needs_test_rows(self):
        panel, protocol, proposals = fixture()
        past = panel.loc[panel.date < pd.Timestamp(protocol.test_start)].copy()
        screen, test, counts = run_lab(past, proposals, protocol, evaluate_test=False)
        self.assertTrue(test.empty)
        self.assertEqual(counts['independent_test']['rows'], 0)
        self.assertTrue(bool(screen.accepted_before_independent_test.iloc[0]))
        full_screen, _, _ = run_lab(panel, proposals, protocol)
        pd.testing.assert_frame_equal(screen, full_screen)

    def test_local_llm_contract_contains_names_only(self):
        panel, _, proposals = fixture()
        names = list(panel.columns)
        self.assertEqual(allowed_inputs(names), ["x_a", "x_b", "x_noise"])
        prompt = build_local_prompt(names)
        self.assertNotIn("label_5", prompt)
        self.assertNotIn("BJ000000", prompt)
        self.assertNotIn("2025-01-02", prompt)
        seen = []
        result = proposal_from_local_llm(lambda text: seen.append(text) or
                                         json.dumps(proposals), names)
        self.assertEqual(result[0]["id"], "interaction_ab")
        self.assertEqual(len(seen), 1)

    def test_reject_code_execution_future_and_label_inputs(self):
        inputs = ["x_a", "x_b"]
        for expression in ("x_a.shift(-1)", "__import__('os')", "x_a[0]",
                           "label_5", "x_a ** 2", "x_a + 1e309"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                parse_proposals([{"id": "bad_expr", "expression": expression,
                                  "hypothesis": "bad"}], inputs)

    def test_maturity_cut_and_future_test_change_do_not_select(self):
        panel, protocol, proposals = fixture()
        train, valid, independent = split_mature(panel, protocol)
        self.assertTrue((train.label_exit_5 < pd.Timestamp(protocol.valid_start)).all())
        self.assertTrue((valid.label_exit_5 < pd.Timestamp(protocol.test_start)).all())
        self.assertTrue((independent.label_exit_5 < pd.Timestamp(protocol.test_end)).all())
        first_screen, first_test, _ = run_lab(panel, proposals, protocol)
        self.assertTrue(bool(first_screen.accepted_before_independent_test.iloc[0]))
        changed = panel.copy()
        changed.loc[changed.date >= pd.Timestamp(protocol.test_start), "label_5"] *= -1
        # These old signal dates are forbidden if the five-day answer was not
        # mature before validation began; changing them must also have no effect.
        changed.loc[(changed.date < pd.Timestamp(protocol.valid_start)) &
                    (changed.label_exit_5 >= pd.Timestamp(protocol.valid_start)), "label_5"] = -1e9
        second_screen, second_test, _ = run_lab(changed, proposals, protocol)
        pd.testing.assert_frame_equal(first_screen, second_screen)
        self.assertAlmostEqual(float(first_test.test_rank_ic_after_train_orientation.iloc[0]),
                               -float(second_test.test_rank_ic_after_train_orientation.iloc[0]),
                               places=10)

    def test_validation_failure_and_duplicate_parent_rejected(self):
        panel, protocol, proposals = fixture()
        changed = panel.copy()
        changed.loc[(changed.date >= pd.Timestamp(protocol.valid_start)) &
                    (changed.date < pd.Timestamp(protocol.test_start)), "label_5"] *= -1
        screen, independent, _ = run_lab(changed, proposals, protocol)
        self.assertFalse(bool(screen.accepted_before_independent_test.iloc[0]))
        self.assertIn("validation_ic_weak_or_reversed", screen.rejection_reasons.iloc[0])
        self.assertTrue(independent.empty)
        direct = [{"id": "direct_a", "expression": "x_a",
                   "hypothesis": "Existing input copy"}]
        screen, _, _ = run_lab(panel, direct, protocol)
        self.assertIn("duplicate_parent_feature", screen.rejection_reasons.iloc[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
