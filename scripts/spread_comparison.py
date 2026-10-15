"""Out-of-sample spread evaluation of identical $100 weekly selections.

Compare moneyline versus point-spread outcomes for the SAME selected games,
sides and stakes. Spread profit uses recorded home/away spread American odds.
nflverse spread_line is expected HOME margin (positive home favorite).
Home cover: (home_score-away_score)-spread_line > 0.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.weighted_slate import simulate_weighted
from scripts.consensus_allocator import returns

def evaluate_spread(predictions):
    report,picks,weekly=simulate_weighted(predictions,weighting='dynamic')
    chosen=picks[picks.stake>0].copy()
    needed=['spread_line','home_spread_odds','away_spread_odds','home_score','away_score']
    baseline=predictions[predictions.model.eq('baseline')][['game_id']+needed]
    selected=chosen.merge(baseline,on='game_id',how='left',validate='one_to_one')
    selected['actual_margin']=selected.home_score-selected.away_score
    selected['home_cover_margin']=selected.actual_margin-selected.spread_line
    selected['spread_side_odds']=np.where(selected.side_home,selected.home_spread_odds,selected.away_spread_odds)
    valid=selected[needed].notna().all(axis=1)&(
       (selected.spread_side_odds<=-100)|(selected.spread_side_odds>=100))
    selected['valid_spread_quote']=valid
    selected['spread_push']=np.isclose(selected.home_cover_margin,0)
    selected['spread_correct']=np.where(selected.side_home,
         selected.home_cover_margin>0,selected.home_cover_margin<0)
    selected['spread_profit']=np.nan
    valid_rows=selected.loc[valid].copy()
    payout=returns(valid_rows.spread_side_odds.to_numpy(float))
    selected.loc[valid,'spread_profit']=np.where(valid_rows.spread_push,0.,
         np.where(valid_rows.spread_correct,valid_rows.stake*payout,-valid_rows.stake))
    selected['ml_profit']=selected.profit
    comparable=selected.loc[valid].copy()
    if comparable.empty:raise ValueError('No valid spread markets to compare')
    totals={}
    for label,column in [('moneyline','ml_profit'),('spread','spread_profit')]:
        amount=float(comparable[column].sum())
        totals[label]={'bets':len(comparable),'staked':float(comparable.stake.sum()),
          'profit':amount,'roi':amount/float(comparable.stake.sum())}
    totals['spread']['covers']=int((comparable.spread_correct&~comparable.spread_push).sum())
    totals['spread']['pushes']=int(comparable.spread_push.sum())
    totals['spread']['losses']=int((~comparable.spread_correct&~comparable.spread_push).sum())
    period=comparable.groupby(['season','week']).agg(spread_profit=('spread_profit','sum'),
      moneyline_profit=('ml_profit','sum'),wagered=('stake','sum')).reset_index()
    season=comparable.groupby('season').agg(spread_profit=('spread_profit','sum'),
      moneyline_profit=('ml_profit','sum'),wagered=('stake','sum')).reset_index()
    summary={'period':'2019-2025','same_game_same_side_same_stakes':True,
      'total_selections':len(selected),'valid_spread_selections':len(comparable),
      'moneyline_vs_spread':totals,
      'spread_profitable_weeks':int((period.spread_profit>0).sum()),
      'moneyline_profitable_weeks_on_same_games':int((period.moneyline_profit>0).sum()),
      'season_summary':season.to_dict('records'),
      'limitations':['Existing model optimizes moneyline winner probability, NOT cover probability.',
        'This is not a spread-optimized forecasting strategy; a real comparison needs a trained score-margin model.',
        'Historical spread odds and lines may not be executable at time of model prediction.',
        'Pushes return original stake; ties in final score already excluded from upstream win model.',
        'No test-year-specific selection tuning.']}
    return summary,selected,period

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='results/spread_comparison')
    a=p.parse_args()
    report,picks,weekly=evaluate_spread(pd.read_csv(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    picks.to_csv(out/'bet_comparison.csv',index=False)
    weekly.to_csv(out/'weekly_comparison.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
