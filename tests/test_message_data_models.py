from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from quant_message.data import minute_events, quarantine_minutes
from quant_message.models import FAMILIES, ModelBundle, fit
from test_message_execution import T


class DataTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(code="920000", datetime=T.replace(tzinfo=None), open=10.,
            high=10.1, low=9.9, close=10., volume=1000., amount=10000.)])

    def test_minute_only_available_after_close(self):
        event = next(minute_events(self.frame()))
        self.assertEqual(event.at, T + timedelta(minutes=1))
        self.assertEqual(event.available_at, event.at)
        self.assertTrue(event.synthetic)

    def test_bad_units_rejected(self):
        f = self.frame(); f.volume /= 100
        with self.assertRaises(ValueError): list(minute_events(f))
        self.assertEqual(next(minute_events(f, volume_unit="lots100")).volume, 1000)

    def test_missing_rows_quarantine_without_filling_prices(self):
        f = self.frame(); f.loc[1] = f.iloc[0]; f.loc[1, "datetime"] += timedelta(minutes=1)
        f.loc[1, ["open", "high", "low", "close", "volume", "amount"]] = np.nan
        clean, audit = quarantine_minutes(f)
        self.assertEqual(len(clean), 1)
        self.assertEqual(audit.iloc[0].reason, "all_fields_missing")

    def test_duplicates_quarantine_both_copies(self):
        f = self.frame(); clean, audit = quarantine_minutes(pd.concat([f, f]))
        self.assertEqual(len(clean), 0); self.assertEqual(len(audit), 2)
        with self.assertRaises(ValueError): list(minute_events(pd.concat([f, f])))


def training_frame():
    rng = np.random.default_rng(17)
    n = 400
    dates = pd.date_range("2026-01-01", periods=20, tz="Asia/Shanghai").repeat(20)
    x = rng.normal(size=(n, 4))
    f = pd.DataFrame(x, columns=["x0", "x1", "x2", "x3"])
    f["date"] = dates.date
    f["available_at"] = dates + pd.Timedelta(hours=16)
    f["label_available_at"] = dates + pd.Timedelta(days=2, hours=15)
    f["target"] = pd.Series(x[:, 0]).groupby(dates).rank(pct=True).to_numpy() - .5
    for j in range(3): f[f"hmm_p{j}"] = 1/3
    return f


class ModelTests(unittest.TestCase):
    def test_all_eight_families_serialize_predict(self):
        f = training_frame()
        at = pd.Timestamp("2026-01-20T09:00:00+08:00")
        x = f.iloc[:20].copy()
        with tempfile.TemporaryDirectory() as temp:
            for name in FAMILIES:
                with self.subTest(family=name):
                    model = fit(f, ["x0", "x1", "x2", "x3"], name, at, 3)
                    a = model.predict(x, at, x[[f"hmm_p{j}" for j in range(3)]])
                    card = model.save(Path(temp) / name)
                    b = ModelBundle.load_trusted(Path(temp) / name).predict(x, at,
                        x[[f"hmm_p{j}" for j in range(3)]])
                    self.assertTrue(np.isfinite(a).all()); np.testing.assert_array_equal(a, b)
                    self.assertLess(pd.Timestamp(card["labels_end"]), at)

    def test_future_training_mutation_does_not_change_fit(self):
        f = training_frame(); at = pd.Timestamp("2026-01-10T09:00:00+08:00")
        future = f.label_available_at >= at
        mutated=f.copy();mutated.loc[future,["target","x0","x1"]]=1e9
        serving=f.iloc[:20];probabilities=serving[[f'hmm_p{i}' for i in range(3)]]
        for family in FAMILIES:
            with self.subTest(family=family):
                a=fit(f,["x0","x1"],family,at,2)
                b=fit(mutated,["x0","x1"],family,at,2)
                self.assertEqual(a.features,b.features)
                np.testing.assert_array_equal(a.predict(serving,at,probabilities),b.predict(serving,at,probabilities))

    def test_past_training_mutation_changes_fit_positive_control(self):
        f=training_frame();at=pd.Timestamp('2026-01-10T09:00:00+08:00')
        a=fit(f,['x0','x1'],'ridge',at,2)
        f.loc[f.label_available_at<at,'target'] *= -1
        b=fit(f,['x0','x1'],'ridge',at,2)
        self.assertFalse(np.array_equal(a.model.coef_,b.model.coef_))

    def test_future_features_rejected_at_inference(self):
        f = training_frame(); at = pd.Timestamp("2026-01-10T09:00:00+08:00")
        a = fit(f, ["x0", "x1"], "ridge", at, 2)
        with self.assertRaises(ValueError): a.predict(f, at)

    def test_model_digest_tampering_rejected(self):
        f = training_frame(); at = pd.Timestamp("2026-01-10T09:00:00+08:00")
        a = fit(f, ["x0", "x1"], "ridge", at, 2)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "model"; a.save(p)
            with (p / "model.pkl").open("ab") as stream: stream.write(b"bad")
            with self.assertRaises(ValueError): ModelBundle.load_trusted(p)


if __name__ == "__main__": unittest.main()
