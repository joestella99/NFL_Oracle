"""Find two-sided ATS value, including losing teams that cover the points.

Fit residual-scale from STRICTLY earlier seasons. Derive normal-cover
probabilities from independently walk-forward predicted final margin.
This is a research assumption, not a guarantee of calibrated probabilities.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scripts.consensus_allocator import returns

def two_sided_value(frame,first=2021,budget=100.,n=6):
    required={'game_id','season','week','margin_prediction','true_margin','spread_line',
              'home_spread_odds','away_spread_odds'}
    if not required.issubset(frame):raise ValueError(f'Missing {required-set(frame)}')
    if frame.duplicated('game_id').any():raise ValueError('Duplicate games')
    rows=[]
    for year in sorted(frame.season.unique()):
        if year<first:continue
        prior=frame[frame.season<year]
        if prior.season.nunique()<2:continue
        residual=(prior.true_margin-prior.margin_prediction).dropna().to_numpy(float)
        sigma=max(5.,float(np.std(residual,ddof=1)))
        current=frame[frame.season==year].copy()
        z=(current.margin_prediction-current.spread_line)/sigma
        home_cover=ndtr(z);away_cover=1-home_cover
        home_payout=returns(current.home_spread_odds.to_numpy(float))
        away_payout=returns(current.away_spread_odds.to_numpy(float))
        ev_home=home_cover*home_payout-(1-home_cover)
        ev_away=away_cover*away_payout-(1-away_cover)
        home=ev_home>=ev_away
        current['home_cover_prob']=home_cover
        current['away_cover_prob']=away_cover
        current['side_home']=home
        current['side_odds']=np.where(home,current.home_spread_odds,current.away_spread_odds)
        current['side_cover_probability']=np.where(home,home_cover,away_cover)
        current['estimated_ev']=np.maximum(ev_home,ev_away)
        current['prior_residual_sd']=sigma
        current['selected_is_underdog']=np.where(home,current.spread_line<0,current.spread_line>0)
        current['selected_outcome_margin']=np.where(home,current.true_margin,-current.true_margin)
        current['side_cover_margin']=np.where(home,current.true_margin-current.spread_line,
                                               current.spread_line-current.true_margin)
        rows.append(current)
    if not rows:raise ValueError('No eligible holdout data')
    d=pd.concat(rows,ignore_index=True)
    stakes=[]
    for (_, _),g in d.groupby(['season','week']):
        selected=g.sort_values(['estimated_ev','game_id'],ascending=[False,True]).head(n).copy()
        # Force full $100 requested budget, favor larger estimated EV without
        # allowing negative scores to become invalid staking weights.
        weights=np.exp(np.clip(3*(selected.estimated_ev.to_numpy(float)-selected.estimated_ev.max()),-12,0))
        weights/=weights.sum()
        cents=int(round(100*budget));units=np.floor(weights*cents).astype(int)
        units[np.argsort(-(weights*cents-units))[:cents-units.sum()]]+=1
        selected['stake']=units/100.
        selected['push']=np.isclose(selected.side_cover_margin,0.)
        selected['cover']=selected.side_cover_margin>0
        payout=returns(selected.side_odds.to_numpy(float))
        selected['profit']=np.where(selected.push,0.,np.where(selected.cover,selected.stake*payout,-selected.stake))
        selected['lost_game_but_covered']=(selected.selected_outcome_margin<0)&selected.cover
        stakes.append(selected)
    bets=pd.concat(stakes,ignore_index=True)
    def metrics(part):
        stake=float(part.stake.sum())
        return {'bets':len(part),'staked':round(stake,2),'net':round(float(part.profit.sum()),2),
                'roi':float(part.profit.sum()/stake) if stake else None,
                'covers':int(part.cover.sum()),'pushes':int(part.push.sum()),
                'lost_game_but_covered':int(part.lost_game_but_covered.sum())}
    periods={}
    for label,mask in [('development',bets.season<=2022),('later_evaluation',bets.season>=2023)]:
        periods[label]=metrics(bets[mask])
    report={'periods':periods,'overall':metrics(bets),
            'underdog_spreads':metrics(bets[bets.selected_is_underdog]),
            'favorite_spreads':metrics(bets[~bets.selected_is_underdog]),
            'season_results':[dict(season=int(year),**metrics(g)) for year,g in bets.groupby('season')],
            'policy':'Choose better-priced side of spread using normal approximation to predicted margin and previous-season margin-residual variability, six per week, spend $100.',
            'limitations':['Model residuals approximated as Gaussian, not validated for point spread calibration.',
             'Historical nflverse spreads/odds may not have verified prekickoff executable timestamps.',
             'Choosing based on maximum estimated edge can magnify model miscalibration.',
             'No risk-free return or guaranteed profit.']}
    return report,bets,d

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--margin-predictions',required=True)
    p.add_argument('--output',default='results/two_sided_spreads')
    a=p.parse_args()
    report,bets,all_games=two_sided_value(pd.read_csv(a.margin_predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    bets.to_csv(out/'weekly_picks.csv',index=False)
    all_games.to_csv(out/'all_game_value_estimates.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
