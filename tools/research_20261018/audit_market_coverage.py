"""Verify actual stored keys, not a transfer count or a maximum daily count."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training_data/research_20261018/market'
def main():
    records=[];keys={}
    for name in ['factors_c01','prices_raw','prices_hfq']:
        p=pd.read_parquet(BASE/(name+'.parquet'),columns=['date','stockid'])
        record={'file':name,'rows':len(p),'stocks_with_rows':int(p.stockid.nunique()),'days':int(p.date.nunique()),
                'first':str(p.date.min().date()),'last':str(p.date.max().date()),
                'duplicate_keys':int(p.duplicated(['date','stockid']).sum()),
                'market_stock_counts':p.groupby(p.stockid.str[:2]).stockid.nunique().to_dict(),
                'market_row_counts':p.groupby(p.stockid.str[:2]).size().to_dict()}
        if record['duplicate_keys']:raise AssertionError('duplicate price/factor keys')
        records.append(record);keys[name]=p
    factor=keys['factors_c01'];matches={}
    for price in ['prices_raw','prices_hfq']:
        merged=factor.merge(keys[price],on=['date','stockid'],how='left',indicator=True,validate='one_to_one')
        missing=merged[merged['_merge']=='left_only'].drop(columns='_merge')
        missing.to_csv(BASE/('factor_keys_missing_'+price+'.csv'),index=False,encoding='utf-8-sig')
        matches[price]={'missing_rows':len(missing),'missing_by_market':missing.groupby(missing.stockid.str[:2]).size().to_dict()}
    if not factor.stockid.str.startswith('BJ').any():raise AssertionError('BJ factor rows were omitted')
    record={'tables':records,'factor_price_matches':matches,
            'warning':'Broad market coverage is NOT verified historical CSI2000 membership coverage. Raw and adjusted-price records may be absent on halts or at listings; diagnose before a future membership join.',
            'repair':'An earlier parsed factors table omitted BJ although source ZIPs contained BJ; rebuilt from all 670 original ZIPs with the expanded allowlist. Earlier BJ strategy experiments used separate existing research_20261011 inputs and are unaffected.'}
    (BASE/'coverage_key_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
