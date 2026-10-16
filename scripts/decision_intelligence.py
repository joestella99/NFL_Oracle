"""Market-aware decision layer: favorites trigger checks, not automatic wagers.

Precommitted conservative shrinkage toward the sportsbook; one exposure per game.
Research simulation only. No auto-promotion to live betting.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.mixed_market_portfolio import candidates
from scripts.consensus_allocator import returns

def evaluate_markets(universe,win_trigger=.75,ml_shrink=.25,ats_shrink=.25,min_edge=.02):
    d=universe.copy()
    # Normalize each game's market moneyline implied probability, removing two-way vig.
    home=d[(d.kind=='ML')&(d.side=='home')][['game_id','odds']].rename(columns={'odds':'home_ml'})
    away=d[(d.kind=='ML')&(d.side=='away')][['game_id','odds']].rename(columns={'odds':'away_ml'})
    d=d.merge(home,on='game_id',validate='many_to_one').merge(away,on='game_id',validate='many_to_one')
    implied=lambda o:1/(1+returns(o))
    home_raw=implied(d.home_ml);away_raw=implied(d.away_ml)
    home_market=home_raw/(home_raw+away_raw)
    market_side=np.where(d.side.eq('home'),home_market,1-home_market)
    # To prevent ATS lines from manufacturing an exact-probability illusion,
    # spread cover estimates shrink toward a coinflip and the moneyline shrinks
    # toward no-vig market probability.
    baseline=np.where(d.kind.eq('ML'),market_side,.5)
    shrink=np.where(d.kind.eq('ML'),ml_shrink,ats_shrink)
    d['conservative_probability']=baseline+shrink*(d.probability-baseline)
    d['conservative_ev']=d.conservative_probability*d.payout-(1-d.conservative_probability)
    # A strong-favorite trigger is based on raw model WIN probability;
    # the trigger leads to price examination, never a forced wager.
    favorite_ml=d[(d.kind=='ML')].sort_values('probability',ascending=False).drop_duplicates('game_id')
    favorite_ml=favorite_ml[['game_id','side','probability']].rename(columns={'side':'favorite_side','probability':'favorite_win_probability'})
    d=d.merge(favorite_ml,on='game_id',validate='many_to_one')
    d['strong_favorite_review']=d.favorite_win_probability>=win_trigger
    d['against_strong_favorite']=d.strong_favorite_review & d.side.ne(d.favorite_side)
    # Opposing favorite ML is not banned; it needs larger and well-supported edge.
    # ATS underdogs against strong favorites also require higher edge than ordinary bets.
    threshold=min_edge+np.where(d.against_strong_favorite,.04,0.)
    d['qualifies']=d.conservative_ev>=threshold
    d['decision_reason']=np.where(~d.qualifies,'NO_BET_INSUFFICIENT_CONSERVATIVE_EDGE',
             np.where(d.strong_favorite_review & d.kind.eq('ML'),'FAVORITE_ML_PRICE_CHECK_PASSED',
              np.where(d.strong_favorite_review,'SPREAD_PRICE_CHECK_PASSED','VALUE_CHECK_PASSED')))
    return d

def simulate(d,weekly_budget=100,max_bets=6):
    picks=[]
    for (_, _),g in d.groupby(['season','week']):
        q=g[g.qualifies].sort_values(['conservative_ev','game_id','kind','side'],
           ascending=[False,True,True,True]).drop_duplicates('game_id').head(max_bets).copy()
        if q.empty:continue
        # Do NOT force a full weekly spend if insufficient edges exist.
        stake=weekly_budget/max_bets
        q['stake']=stake
        q['profit']=np.where(q.push,0,np.where(q.won,q.stake*q.payout,-q.stake))
        picks.append(q)
    if not picks:
        raise ValueError('No selections passed research policy.')
    return pd.concat(picks,ignore_index=True)

def experiment(winners,margins):
    universe=candidates(margins,winners)
    ranked=evaluate_markets(universe)
    bets=simulate(ranked)
    results={}
    for label,subset in [('development',bets[bets.season<=2022]),
                         ('later_evaluation',bets[bets.season>=2023]),('all',bets)]:
        amount=float(subset.stake.sum())
        results[label]={'bets':len(subset),'staked':round(amount,2),'net':round(float(subset.profit.sum()),2),
          'roi':float(subset.profit.sum()/amount) if amount else None,
          'moneyline_bets':int(subset.kind.eq('ML').sum()),
          'spread_bets':int(subset.kind.eq('ATS').sum()),
          'strong_favorite_reviews_selected':int(subset.strong_favorite_review.sum())}
    return {'results':results,'rules':{'strong_favorite_review_threshold':.75,'ml_market_shrinkage':.25,
      'spread_shrinkage_to_50_50':.25,'minimum_estimated_edge':.02,
      'extra_opposing_favorite_edge':.04,'six_bet_cap':6,'fixed_stake':100/6},
      'caveats':['Rule values are exploratory, chosen after seeing historical results: not virgin holdout.',
        'ATS probabilities omit point-spread push mass; conservative shrinking does not itself establish calibration.',
        'Historical lines are not timestamp-verified executable quotes.',
        'Same-side ML and ATS are correlated exposures, NOT a hedge.',
        'This no-forced-bets strategy has different weekly exposure than $100 forced spend.']},bets,ranked

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--winner-predictions',required=True)
    ap.add_argument('--margin-predictions',required=True)
    ap.add_argument('--output',default='results/decision_intelligence')
    args=ap.parse_args()
    report,picks,universe=experiment(pd.read_csv(args.winner_predictions),pd.read_csv(args.margin_predictions))
    path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    (path/'report.json').write_text(json.dumps(report,indent=2))
    picks.to_csv(path/'bets.csv',index=False)
    universe.to_csv(path/'market_checks.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
