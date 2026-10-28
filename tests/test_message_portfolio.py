from datetime import datetime
import unittest

import pandas as pd

from quant_message.domain import Costs, Instrument, SH
from quant_message.paper import PaperBroker
from quant_message.portfolio_audit import reconstruct


def fill(identity, symbol='A', side='buy', quantity=100, price=10, day='2026-09-23'):
    return dict(id=identity,order_id=identity,symbol=symbol,side=side,quantity=quantity,
                price=price,fee=Costs().total(quantity*price,side),at=day+'T10:00:00+08:00')


class PortfolioAuditTests(unittest.TestCase):
    def marks(self):
        return pd.DataFrame([dict(date='2026-09-23',symbol='A',price=10),
                             dict(date='2026-09-24',symbol='A',price=9)])

    def test_dividend_is_entitlement_not_buying_power(self):
        action=dict(date='2026-09-24',symbol='A',dividend=1,ratio=1)
        f,m=reconstruct(pd.DataFrame([fill('1')]),self.marks(),[action],slippage=0)
        self.assertAlmostEqual(f.cash.iloc[0],f.cash.iloc[1])
        self.assertAlmostEqual(f.audited_equity.iloc[0],f.audited_equity.iloc[1])
        self.assertEqual(m['dividend_receivable'],100)
        b=PaperBroker([Instrument('A.BSE')]);b.position('A.BSE').quantity=100
        b.corporate_action('x','A.BSE',datetime(2026,9,24,9,tzinfo=SH),1,credit_cash=False)
        b.corporate_action('x','A.BSE',datetime(2026,9,24,9,tzinfo=SH),1,credit_cash=False)
        self.assertEqual(b.cash,b.initial)
        self.assertEqual(b.dividend_receivable,100)

    def test_multisymbol_sparse_calendar_carries_prior_mark(self):
        marks=pd.concat([self.marks(),pd.DataFrame([dict(date='2026-09-23',symbol='B',price=20)])])
        f,m=reconstruct(pd.DataFrame([fill('1'),fill('2',symbol='B',price=20)]),marks)
        self.assertEqual(len(f),2)
        self.assertEqual(f.stale_held_symbols.iloc[-1],1)
        self.assertEqual(f.market_value.iloc[-1],2900)
        self.assertFalse(f.isna().any().any())

    def test_duplicate_fill_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            reconstruct(pd.DataFrame([fill('1'),fill('1')]),self.marks())

    def test_t1_sale_rejected(self):
        with self.assertRaisesRegex(ValueError,'T\\+1'):
            reconstruct(pd.DataFrame([fill('1'),fill('2',side='sell')]),self.marks())

    def test_fee_tampering_rejected(self):
        f=fill('1');f['fee']=0
        with self.assertRaisesRegex(ValueError,'fee'):
            reconstruct(pd.DataFrame([f]),self.marks())


if __name__=='__main__':unittest.main()
