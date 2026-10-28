from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd

from quant_message.models import fit
from quant_message.production import make_signal, publish, refit_due
from quant_message.data import read_signal, read_jsonl
from test_message_data_models import training_frame


class ProductionTests(unittest.TestCase):
    def test_refit_boundaries(self):
        self.assertTrue(refit_due('2026-04-01T09:00:00+08:00', '2026-03-31T17:00:00+08:00', 'quarterly'))
        self.assertFalse(refit_due('2026-04-02T09:00:00+08:00', '2026-04-01T17:00:00+08:00', 'quarterly'))
        self.assertTrue(refit_due('2026-04-02T09:00:00+08:00', '2026-04-01T17:00:00+08:00', 'daily'))
        self.assertFalse(refit_due('2026-04-02T09:00:00+08:00', '2026-04-01T17:00:00+08:00', 'monthly'))

    def test_timestamped_fit_to_packet_roundtrip(self):
        f = training_frame(); at = '2026-01-20T17:00:00+08:00'
        b = fit(f, ['x0','x1'], 'ridge', at, 2)
        f = f.iloc[:10].copy()
        f['symbol'] = [f'920{i:03d}.BSE' for i in range(10)]
        f['reference']=10.;f['volatility']=.2;f['industry']='industry';f['size_group']='small'
        f['pool_rank']=.5;f['rebound']=True
        m=dict(observed_at='2026-01-20T15:00:00+08:00',available_at=at,market_cap=.5,market_vol=.2,model_id='ridge')
        s=make_signal(b,f,m,at,'2026-01-21T09:30:00+08:00','2026-01-21T14:57:00+08:00','packet1')
        with TemporaryDirectory() as d:
            p=Path(d)/'signal.jsonl';publish(s,p)
            loaded=read_signal(next(read_jsonl(p)));loaded.validate(loaded.valid_from)
            self.assertTrue(loaded.research_only)
            self.assertEqual(len(loaded.candidates),10)
        m['available_at']='2026-01-22T00:00:00+08:00'
        with self.assertRaises(ValueError):make_signal(b,f,m,at,s.valid_from,s.expires_at,'bad')


if __name__=='__main__':unittest.main()
