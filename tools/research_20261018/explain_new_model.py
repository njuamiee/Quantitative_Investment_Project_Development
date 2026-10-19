"""Native tree contributions: explain a frozen prediction, not realized P&L."""
from pathlib import Path
import json
import joblib,numpy as np,pandas as pd
from factor_loop_experiment import engineered
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training_runs/research_20261018/factor_loop'

def main():
    f=json.loads((RUN/'frozen_selection.json').read_text(encoding='utf-8'))
    fitted=joblib.load(RUN/'augmented.joblib')
    raw=pd.read_parquet(ROOT/'training_data/research_20261011/model_panel.parquet',filters=[('date','>=',pd.Timestamp('2026-01-01')),('date','<',pd.Timestamp('2026-10-01'))])
    p=engineered(raw,f['chosen'])
    sample=p[p.date.isin([p.date.min(),p.date.max()])].copy()
    x=sample[fitted['factors']].fillna(0)
    c=fitted['model'].predict(x,pred_contrib=True)
    prediction=fitted['model'].predict(x)
    gap=float(np.max(np.abs(c.sum(axis=1)-prediction)))
    if gap>1e-8:raise AssertionError('contribution sum mismatch')
    score=sample[['date','stockid']].copy();score['model_score']=prediction;score['base_score']=c[:,-1]
    score=pd.concat([score,pd.DataFrame(c[:,:-1],index=sample.index,
                    columns=['contribution__'+col for col in fitted['factors']])],axis=1)
    score.to_csv(RUN/'example_prediction_contributions.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for day,g in score.groupby('date'):
        top=g.sort_values(['model_score','stockid'],ascending=[False,True]).iloc[0]
        details=sorted([{'factor':col,'contribution':float(top['contribution__'+col]),
                         'input_value':float(x.loc[top.name,col])}
                        for col in fitted['factors']],key=lambda r:abs(r['contribution']),reverse=True)
        rows.append({'date':str(day.date()),'stockid':top.stockid,'base_score':top.base_score,
                     'model_score':top.model_score,'top_contributions':details[:8],
                     'all_contributions_sum':float(sum(r['contribution'] for r in details))})
    (RUN/'prediction_explanation.json').write_text(json.dumps({'max_sum_error':gap,'examples':rows,
        'meaning':'Contributions to predicted within-day rank score; neither future return percentages nor causal profit explanations.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXPLANATION_COMPLETE',len(score),gap)

if __name__=='__main__':main()
