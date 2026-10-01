"""Exact pairwise Spearman correlation, grouping columns with identical missingness.

Ranks are recomputed on each pair's common observations, exactly as pandas does.
This is a computational optimization, not a change to factor selection.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def daily_spearman(frame,min_periods=30):
    x=frame.to_numpy(dtype=float);finite=np.isfinite(x);groups={}
    for j,mask in enumerate(finite.T):groups.setdefault(np.packbits(mask).tobytes(),[]).append(j)
    groups=list(groups.values());out=np.full((x.shape[1],x.shape[1]),np.nan)
    for i,left in enumerate(groups):
        for right in groups[i:]:
            mask=finite[:,left[0]]&finite[:,right[0]]
            if mask.sum()<min_periods:continue
            a=rankdata(x[np.ix_(mask,left)],axis=0);b=rankdata(x[np.ix_(mask,right)],axis=0)
            a-=a.mean(axis=0);b-=b.mean(axis=0)
            denom=np.sqrt((a*a).sum(axis=0)[:,None]*(b*b).sum(axis=0)[None,:])
            with np.errstate(invalid='ignore',divide='ignore'):corr=a.T@b/denom
            out[np.ix_(left,right)]=corr;out[np.ix_(right,left)]=corr.T
    return pd.DataFrame(out,index=frame.columns,columns=frame.columns)
