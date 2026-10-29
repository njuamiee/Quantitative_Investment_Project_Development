from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from quant_message.database_io import connect_database,digest_ticks
from quant_message.domain import SH


def tick(at):
    t=SimpleNamespace(vt_symbol='920000.BSE',datetime=at,last_price=10.,volume=100.,
                      turnover=1000.,pre_close=9.,limit_up=13.,limit_down=7.)
    for side in ('bid','ask'):
        for what in ('price','volume'):
            for i in range(1,6):setattr(t,f'{side}_{what}_{i}',10. if what=='price' else 100.)
    return t


class DatabaseContractTests(unittest.TestCase):
    def test_digest_normalizes_timezone_and_source_order(self):
        local=datetime(2026,9,23,9,30,tzinfo=SH)
        a=tick(local);b=tick(local.replace(hour=10))
        self.assertEqual(digest_ticks([a,b]),digest_ticks([b,tick(local.astimezone(timezone.utc))]))

    def test_digest_detects_deep_book_change(self):
        a=tick(datetime(2026,9,23,9,30,tzinfo=SH));b=tick(a.datetime)
        b.ask_volume_5+=1
        self.assertNotEqual(digest_ticks([a]),digest_ticks([b]))

    def test_original_or_arbitrary_database_names_rejected(self):
        for name in ('vnpy','quant_message_20261029_base;DROP DATABASE vnpy','another_project'):
            with self.assertRaises(ValueError):connect_database(name,create=True)


if __name__=='__main__':unittest.main()
