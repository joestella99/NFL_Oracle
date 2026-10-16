"""Research alternative calibrated probability / offered-odds six-pick strategies.

Fits a probability combiner exclusively on previous years, with one year's
outcomes never entering that year's calibrator. Compare a fixed development
period (2019-22) to a later evaluation period (2023-25).
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

VARIANTS=('raw_edge','disagreement_penalized_edge','calibrated_favorite','calibrated_favorite_price')
def matrix(df):
    p=lambda key: np.clip(df[key].to_numpy(float),.0001,.9999)
    market=logit(p('market_home_prob'))
    return np.column_stack([market]+[logit(p(k))-market for k in ('baseline','matchup','sustainability')])

def prepare(raw, first_year=2019):
    if raw.duplicated(['game_id','model']).any():raise ValueError('Duplicate forecasts')
    base=raw[raw.model.eq('baseline')][['game_id','season','week','home_win','market_home_prob','home_moneyline','away_moneyline']]
    wide=raw.pivot(index='game_id',columns='model',values='prediction').reset_index()
    d=base.merge(wide,on='game_id',validate='one_to_one')
    d['calibrated']=np.nan
    for year in sorted(d.season.unique()):
        if year<first_year:continue
        train=d[d.season<year];test=d[d.season==year]
        if train.season.nunique()<2:continue
        model=LogisticRegression(C=.05,max_iter=1000).fit(matrix(train),train.home_win)
        d.loc[test.index,'calibrated']=model.predict_proba(matrix(test))[:,1]
    return d[d.calibrated.notna()].copy()

def odds_returns(odds):
    odds=np.asarray(odds,float)
    if not np.isfinite(odds).all() or np.any((odds>-100)&(odds<100)):raise ValueError('Invalid moneyline')
    return np.where(odds>0,odds/100,100/-odds)

def slate(df,variant,n=6,budget=100):
    d=df.copy()
    p=d.calibrated.to_numpy(float)
    bh=odds_returns(d.home_moneyline);ba=odds_returns(d.away_moneyline)
    ev_home=p*bh-(1-p);ev_away=(1-p)*ba-p
    if variant in ('raw_edge','disagreement_penalized_edge'):
        side=ev_home>=ev_away
        score=np.maximum(ev_home,ev_away)
        if variant=='disagreement_penalized_edge':
            score=score-.03*np.abs(logit(np.clip(d.baseline,.0001,.9999))-logit(np.clip(d.market_home_prob,.0001,.9999)))
        wager_weight=lambda values: np.exp(np.clip((values-values.max())*5,-20,0))
    elif variant in ('calibrated_favorite','calibrated_favorite_price'):
        side=p>=.5
        score=np.maximum(p,1-p)
        if variant=='calibrated_favorite_price': score=score*(1+np.where(side,ev_home,ev_away))
        wager_weight=lambda values: np.maximum(.01,values-.5)**2
    else:raise ValueError(variant)
    d['score']=score
    d['chosen_home']=side
    d['unit_profit']=np.where(side,np.where(d.home_win==1,bh,-1),np.where(d.home_win==0,ba,-1))
    picked=[]
    for (year,week),g in d.groupby(['season','week']):
        selected=g.sort_values(['score','game_id'],ascending=[False,True]).head(n).copy()
        weight=wager_weight(selected.score.to_numpy(float))
        weight=weight/weight.sum()
        selected['stake']=weight*budget
        selected['profit']=selected.stake*selected.unit_profit
        picked.append(selected)
    return pd.concat(picked,ignore_index=True)

def evaluate(raw):
    prepared=prepare(raw)
    by_period=[];selections=[]
    for name in VARIANTS:
        data=slate(prepared,name)
        data['strategy']=name
        selections.append(data)
        for period,mask in [('development',data.season.between(2019,2022)),
                            ('later_evaluation',data.season.between(2023,2025))]:
            sub=data[mask]
            weekly=sub.groupby(['season','week']).profit.sum()
            by_period.append(dict(strategy=name,period=period,weeks=len(weekly),
                bets=len(sub),staked=round(float(sub.stake.sum()),2),
                profit=round(float(sub.profit.sum()),2),
                roi=float(sub.profit.sum()/sub.stake.sum()),
                profitable_weeks=int((weekly>0).sum())))
    return dict(policy='Six bets per week, $100 allocated with fixed predeclared score formulas. Calibrator fit on prior seasons only.',
                warning='A posteriori research. Development results cannot be used as evidence of a reliable betting edge.',
                results=by_period),pd.concat(selections,ignore_index=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='results/odds_aware_alternatives')
    a=p.parse_args()
    report,selections=evaluate(pd.read_csv(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(report['results']).to_csv(out/'comparison.csv',index=False)
    selections.to_csv(out/'picks.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
