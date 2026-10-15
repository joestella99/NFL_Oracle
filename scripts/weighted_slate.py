"""Forced-spend $100 weighted slate experiment, distinct from the no-bet EV policy.
Picks are ranked by a blend of conservative confidence and price-adjusted edge.
The full budget is allocated in integer cents over at most 6 games; no guarantees.
Odds are historical recorded prices, not necessarily available before kickoff.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.consensus_allocator import assemble,returns

def distribute(group,budget=100.0,max_bets=6,confidence_weight=.65):
    if not 0<=confidence_weight<=1 or budget<=0 or max_bets<1:raise ValueError('Invalid allocation policy')
    d=group.copy()
    d['stake']=0.
    odds=d.selected_odds.to_numpy(float)
    b=returns(odds)
    p=d.minimum_side_prob.to_numpy(float)
    # Three correlated models; choose market favorite with the smallest estimate.
    # The price-adjusted edge can be negative: forced spending is not evidence of value.
    edge=p-(1/(1+b))
    d['estimated_edge']=edge
    d['confidence_floor']=p
    score=confidence_weight*p+(1-confidence_weight)*edge
    # Penalize disagreement, but do not exclude every game on a no-edge slate.
    score=score-np.where(d.unanimous.to_numpy(bool),0.,.10)
    d['selection_score']=score
    selected=d.sort_values(['selection_score','game_id'],ascending=[False,True]).head(max_bets)
    # Positive weights and modest floor prevent one single game from absorbing the slate.
    raw=np.maximum(.001,selected.selection_score.to_numpy(float)-selected.selection_score.min()+.1)
    fractions=raw/raw.sum()
    cents=round(budget*100)
    units=np.floor(fractions*cents).astype(int)
    leftover=cents-units.sum()
    order=np.argsort(-(fractions*cents-units),kind='stable')
    units[order[:leftover]]+=1
    d.loc[selected.index,'stake']=units/100.
    correct=(d.side_home.to_numpy(bool)==d.home_win.to_numpy(int))
    d['profit']=np.where(correct,d.stake*returns(d.selected_odds.to_numpy(float)),-d.stake)
    return d

def simulate_weighted(frame,budget=100.,first=2019):
    games=assemble(frame)
    results=[distribute(g,budget) for _,g in games.loc[games.season>=first].groupby(['season','week'])]
    if not results:raise ValueError('No games')
    d=pd.concat(results,ignore_index=True)
    weekly=d.groupby(['season','week']).agg(wagers=('stake',lambda x:int((x>0).sum())),
       staked=('stake','sum'),profit=('profit','sum')).reset_index()
    weekly['winning_week']=weekly.profit>0
    assert np.allclose(weekly.staked.to_numpy(float),budget,atol=.005)
    by_season=weekly.groupby('season').agg(weeks=('week','size'),profitable_weeks=('winning_week','sum'),
       total_profit=('profit','sum'),total_staked=('staked','sum')).reset_index()
    report=dict(weeks=len(weekly),bets=int((d.stake>0).sum()),total_staked=float(d.stake.sum()),
       total_profit_proxy=float(d.profit.sum()),roi_proxy=float(d.profit.sum()/d.stake.sum()),
       profitable_weeks=int(weekly.winning_week.sum()),losing_or_break_even_weeks=int((~weekly.winning_week).sum()),
       season_results=by_season.to_dict('records'),policy='Spend full $100 on 6 ranked picks each week, weighted by conservative model-confidence and price edge.',
       limitations='Forced spending may lead to negative expected value. Betting profits are not guaranteed. Historical recorded moneylines are not verified executable quotes; policy developed after seeing historical outcomes.')
    return report,d,weekly

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='results/weighted_slate_20')
    a=p.parse_args()
    report,d,weekly=simulate_weighted(pd.read_csv(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    d.to_csv(out/'picks.csv',index=False)
    weekly.to_csv(out/'weekly.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
