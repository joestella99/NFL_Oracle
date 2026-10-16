"""Combined moneyline and ATS portfolio experiment; one market per game.

Win probabilities are chronological Oracle outputs; spread cover probabilities
come from an earlier-season-only residual model. Stakes are distributed over six
unique games. The market may have negative estimated edge, because the requested
$100 budget is spent in full. Historical quotes are not timestamp-verified.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scripts.consensus_allocator import returns
from scripts.odds_aware_alternatives import prepare

def candidates(margins,winners,first=2021):
    w=prepare(winners)
    merged=margins.merge(w[['game_id','calibrated','home_moneyline','away_moneyline']],on='game_id',validate='one_to_one')
    rows=[]
    for year in sorted(merged.season.unique()):
        if year<first:continue
        prior=margins.loc[margins.season<year]
        if prior.season.nunique()<2:continue
        residual=prior.true_margin-prior.margin_prediction
        sigma=max(5.,float(residual.std()))
        d=merged.loc[merged.season==year]
        for r in d.itertuples(index=False):
            if pd.isna(r.spread_line):continue
            p_home=float(r.calibrated)
            cover_home=float(ndtr((r.margin_prediction-r.spread_line)/sigma))
            for kind,side,prob,odds in [
                ('ML','home',p_home,r.home_moneyline),
                ('ML','away',1-p_home,r.away_moneyline),
                ('ATS','home',cover_home,r.home_spread_odds),
                ('ATS','away',1-cover_home,r.away_spread_odds)]:
                payout=float(returns([odds])[0])
                ev=prob*payout-(1-prob)
                if kind=='ML':
                    outcome=(r.true_margin>0) if side=='home' else (r.true_margin<0)
                    push=False
                else:
                    cover_margin=r.true_margin-r.spread_line
                    outcome=cover_margin>0 if side=='home' else cover_margin<0
                    push=bool(np.isclose(cover_margin,0))
                rows.append({'game_id':r.game_id,'season':year,'week':r.week,'kind':kind,'side':side,
                             'probability':prob,'odds':odds,'payout':payout,'estimated_ev':ev,
                             'won':bool(outcome),'push':push,
                             'lost_outside_spread':bool(kind=='ATS' and outcome and
                                 ((r.true_margin<0 and side=='home') or (r.true_margin>0 and side=='away')))})
    return pd.DataFrame(rows)

def allocate(frame,mode='mixed',budget=100,n=6):
    if mode not in ('mixed','ml_only','ats_only'):raise ValueError(mode)
    d=frame if mode=='mixed' else frame.loc[frame.kind.eq('ML' if mode=='ml_only' else 'ATS')]
    picks=[]
    for _,g in d.groupby(['season','week']):
        # Strongest estimated value per game, prohibit correlated wagers on same game.
        best=g.sort_values(['estimated_ev','game_id','kind','side'],ascending=[False,True,True,True])
        best=best.drop_duplicates('game_id').head(n).copy()
        if len(best)<n:continue
        strength=best.estimated_ev.to_numpy(float)
        weights=np.exp(np.clip(3*(strength-strength.max()),-12,0));weights/=weights.sum()
        cents=round(budget*100)
        units=np.floor(weights*cents).astype(int)
        units[np.argsort(-(weights*cents-units))[:cents-units.sum()]]+=1
        best['stake']=units/100.
        best['profit']=np.where(best.push,0,np.where(best.won,best.stake*best.payout,-best.stake))
        picks.append(best)
    if not picks:raise ValueError('No eligible slates')
    return pd.concat(picks,ignore_index=True)

def experiment(margins,winners):
    universe=candidates(margins,winners)
    results={};all_picks=[]
    for mode in ('mixed','ml_only','ats_only'):
        picks=allocate(universe,mode)
        picks['strategy']=mode;all_picks.append(picks)
        summary=[]
        for label,sub in [('development',picks[picks.season<=2022]),
                          ('later_evaluation',picks[picks.season>=2023]),('all',picks)]:
            stake=float(sub.stake.sum());weeks=sub.groupby(['season','week']).profit.sum()
            summary.append({'period':label,'bets':len(sub),'weeks':len(weeks),
                            'profit':round(float(sub.profit.sum()),2),'staked':round(stake,2),
                            'roi':float(sub.profit.sum()/stake) if stake else None,
                            'profitable_weeks':int((weeks>0).sum()),
                            'ml_bets':int((sub.kind=='ML').sum()),'spread_bets':int((sub.kind=='ATS').sum()),
                            'losing_teams_covering':int(sub.lost_outside_spread.sum())})
        results[mode]=summary
    return {'results':results,'policy':'Four market outcomes considered per game; six distinct games per week; all $100 spent using fixed exponential estimated-EV weights.',
      'limitations':['Margin cover probability uses normal residual assumption and ignores push mass when evaluating edge.',
       'All market prices are retrospective and not timestamp-verified.',
       '2023-25 is later in time but not an untouched holdout: features and selectors were developed after viewing historical outcomes.',
       'Selecting highest estimated edge can amplify predictive model misspecification.']},pd.concat(all_picks,ignore_index=True),universe

def main():
    p=argparse.ArgumentParser();p.add_argument('--winner-predictions',required=True)
    p.add_argument('--margin-predictions',required=True)
    p.add_argument('--output',default='results/mixed_market_portfolio')
    a=p.parse_args()
    report,picks,candidates_df=experiment(pd.read_csv(a.margin_predictions),pd.read_csv(a.winner_predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    picks.to_csv(out/'selected_bets.csv',index=False)
    candidates_df.to_csv(out/'candidate_bets.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
