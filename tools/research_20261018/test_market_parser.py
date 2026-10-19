import unittest,tempfile,zipfile
from pathlib import Path
import pandas as pd
from sync_market import factor_frame,FACTOR_COLUMNS
class MarketParserTests(unittest.TestCase):
    def test_keeps_old_new_bj_codes_and_both_other_markets(self):
        codes=['BJ430017','BJ830799','BJ920799','SH600000','SZ000001','SH000001']
        df=pd.DataFrame({'stockid':codes,'enddate':[45811]*len(codes),**{c:[1.]*len(codes) for c in FACTOR_COLUMNS}})
        day=str((pd.Timestamp('1899-12-30')+pd.Timedelta(days=45811)).date())
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'f.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('f.csv',df.to_csv(index=False))
            out=factor_frame(p,day)
        self.assertEqual(set(out.stockid),set(codes[:-1]))
        self.assertEqual(int(out.stockid.str.startswith('BJ').sum()),3)
if __name__=='__main__':unittest.main()
