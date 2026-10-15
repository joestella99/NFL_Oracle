"""Research-only consensus and $20 slate allocator.

Consensus requires market + three Oracle variants to agree, not four independent
votes: all three Oracle variants share market features and are correlated.
Allocation uses a conservative lower probability bound and a capped fractional
Kelly budget. No candidate -> keep all cash. Recorded historical odds are not
verified executable offers.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=('baseline','matchup','sustainability')

def assemble(frame):
    required={'game_id','season','week','model','prediction','market_home_prob',
              'home_win','home_moneyline','away_moneyline'}
    if not required.issubset(frame):raise ValueError(f'Missing {sorted(required-set(frame))}')
    if frame.duplicated(['game_id','model']).any():raise ValueError('Duplicate game/model')
    base=frame[frame.model==MODELS[0]][['game_id','season','week','home_win',
             'market_home_prob','home_moneyline','away_moneyline']].copy()
    for model in MODELS:
        sub=frame[frame.model==model][['game_id','prediction']].rename(columns={'prediction':model+'_prob'})
        base=base.merge(sub,on='game_id',how='inner',validate='one_to_one')
    if len(base)*3!=len(frame):raise ValueError('Incomplete or unexpected model memberships')
    columns=[m+'_prob' for m in MODELS]
    all_p=base[columns].to_numpy(float)
    market=base.market_home_prob.to_numpy(float)
    agreement=((all_p>=.5)==(market[:,None]>=.5)).all(axis=1)
    base['unanimous']=agreement
    base['consensus_prob']=np.mean(all_p,axis=1)
    base['minimum_side_prob']=np.where(market>=.5,
          np.min(all_p,axis=1),1-np.max(all_p,axis=1))
    base['market_side_prob']=np.maximum(market,1-market)
    base['side_home']=market>=.5
    base['selected_odds']=np.where(base.side_home,base.home_moneyline,base.away_moneyline)
    return base

def returns(odds):
    odds=np.asarray(odds,float)
    if np.any(~np.isfinite(odds)) or np.any((odds>-100)&(odds<100)):
        raise ValueError('Invalid odds')
    return np.where(odds>0,odds/100,100/(-odds))

def allocate(group,budget=20.,min_confidence=.70,min_edge=.04,max_bets=4,max_each=8.):
    if budget<=0 or max_each<=0 or max_bets<1:raise ValueError('Invalid budget policy')
    d=group.copy()
    d['stake']=0.
    d['ev_lower']=np.nan
    b=returns(d.selected_odds.to_numpy(float))
    # Conservative stress haircut to account for model correlation and calibration risk:
    # do NOT interpret unanimous model probabilities as independent evidence.
    prob=np.maximum(.0,d.minimum_side_prob.to_numpy(float)-.03)
    raw_ev=prob*b-(1-prob)
    market_raw=1/(1+b)
    d['ev_lower']=raw_ev
    d['lower_prob']=prob
    eligible=d.unanimous.to_numpy(bool)&(prob>=min_confidence)&((prob-market_raw)>=min_edge)&(raw_ev>0)
    chosen=d.loc[eligible].sort_values('ev_lower',ascending=False).head(max_bets).copy()
    if len(chosen):
        b=returns(chosen.selected_odds.to_numpy(float))
        p=chosen.lower_prob.to_numpy(float)
        kelly=np.maximum(0.,(p*b-(1-p))/b)
        # Quarter Kelly of fixed $20 slate, with per-game cap and cash retention.
        stakes=np.minimum(max_each,budget*.25*kelly)
        if stakes.sum()>budget:stakes=stakes*budget/stakes.sum()
        d.loc[chosen.index,'stake']=np.floor(stakes*100)/100
    actual=d.home_win.to_numpy(int)
    correct=(d.side_home.to_numpy(bool)==(actual==1))
    d['profit']=np.where(correct,d.stake*returns(d.selected_odds.to_numpy(float)),-d.stake)
    return d

def simulate(frame,first=2019):
    base=assemble(frame)
    # All policy constants specified before evaluating outcomes; no per-year tuning.
    parts=[]
    for (year,week),g in base.loc[base.season>=first].groupby(['season','week']):
        parts.append(allocate(g))
    d=pd.concat(parts,ignore_index=True)
    weekly=d.groupby(['season','week']).agg(
      games=('game_id','size'),wagers=('stake',lambda x:int((x>0).sum())),
      staked=('stake','sum'),profit=('profit','sum')).reset_index()
    weekly['unspent']=20.-weekly.staked
    report={'first_season':first,'last_season':int(d.season.max()),
            'games':len(d),'unanimous_games':int(d.unanimous.sum()),
            'bets':int((d.stake>0).sum()),'slates':len(weekly),
            'total_staked':float(d.stake.sum()),'net_profit_proxy':float(d.profit.sum()),
            'roi_on_staked':float(d.profit.sum()/d.stake.sum()) if d.stake.sum()>0 else None,
            'average_weekly_spend':float(weekly.staked.mean()),
            'policy':'Unanimous market plus 3 correlated Oracle models; minimum conservative side confidence 70%; 4pp edge vs raw offered odds; 3pp haircut; quarter-Kelly with $8 cap, at most 4 bets per week, cash held otherwise.',
            'warnings':['Policy is experimental and these seasons are retrospective, not untouched validation.',
              'Historical odds may not be executable before kickoff; reported returns are proxies.',
              'All three Oracle models include sportsbook probability; consensus is not independent evidence.',
              'The $20 is a maximum budget, not a target to spend.']}
    return report,d,weekly

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='results/consensus_20')
    a=p.parse_args()
    report,d,weekly=simulate(pd.read_csv(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    d.to_csv(out/'all_games.csv',index=False)
    weekly.to_csv(out/'weekly_slates.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
