"""Miss diagnostics: which PRE-GAME football signal pointed toward the winning side?

Join fixed mixed-market selections to walk-forward features, examine feature
orientation toward selected team, and compare predictive direction on 2021-22
versus 2023-25. No outcomes used to construct feature values.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from nfl_oracle.advanced import ADVANCED_FEATURES
from nfl_oracle.matchups import MATCHUP_FEATURES
from nfl_oracle.stability import STABILITY_FEATURES

FEATURES=ADVANCED_FEATURES+MATCHUP_FEATURES+STABILITY_FEATURES

def audit(predictions,selections):
    base=predictions[predictions.model.eq('baseline')][['game_id']+FEATURES]
    if base.duplicated('game_id').any():raise ValueError('Duplicate source games')
    picks=selections[selections.strategy.eq('mixed')].copy()
    if picks.empty:raise ValueError('No mixed-market selections')
    d=picks.merge(base,on='game_id',validate='many_to_one')
    # + feature means better for HOME team, orient signs to PICKED team.
    flip=np.where(d.side.eq('home'),1.,-1.)
    d['correct']=d.won.astype(bool)
    d['missed']=~d.correct & ~d.push.astype(bool)
    d['true_win_side']=np.where(d.kind.eq('ATS'),
       np.where(d.side.eq('home'),d.won,~d.won),
       np.where(d.side.eq('home'),d.won,~d.won))
    rows=[]
    for kind in ('ML','ATS','ALL'):
        k=d if kind=='ALL' else d[d.kind.eq(kind)]
        for feature in FEATURES:
            valid=k[[feature,'correct','season','side']].dropna().copy()
            if valid.empty:continue
            orientation=np.where(valid.side.eq('home'),1.,-1.)
            selected_adv=valid[feature].to_numpy(float)*orientation
            # Compare the pregame selected-team advantage (negative = opponent better)
            # against actual outcome. Larger advantage should imply greater chance pick wins.
            for period,mask in [('development',valid.season.between(2021,2022)),
                                ('later_evaluation',valid.season.between(2023,2025)),
                                ('all',valid.season.between(2021,2025))]:
                outcomes=valid.loc[mask,'correct'].astype(int).to_numpy()
                values=selected_adv[mask.to_numpy()]
                if len(np.unique(outcomes))<2:continue
                wins=values[outcomes==1];losses=values[outcomes==0]
                auc=roc_auc_score(outcomes,values)
                rows.append(dict(kind=kind,feature=feature,period=period,
                    n=len(outcomes),misses=int((outcomes==0).sum()),
                    winning_pick_avg=float(np.mean(wins)),
                    missed_pick_avg=float(np.mean(losses)),
                    delta_win_minus_miss=float(np.mean(wins)-np.mean(losses)),
                    auc=float(auc),
                    fraction_misses_opponent_positive=float((losses<0).mean())))
    stats=pd.DataFrame(rows)
    piv=stats.pivot_table(index=['kind','feature'],columns='period',values='auc').reset_index()
    confirmed=piv[(piv.later_evaluation>.52)&(piv.development>.52)].sort_values('later_evaluation',ascending=False)
    report={'total_selected':int(len(d)),'overall_missed':int(d.missed.sum()),
      'winner_misses':int(d.loc[d.kind.eq('ML'),'missed'].sum()),
      'spread_misses':int(d.loc[d.kind.eq('ATS'),'missed'].sum()),
      'stable_positive_signals':confirmed.to_dict('records'),
      'interpretation':'AUC > .5 means the pregame feature favors selected teams that ultimately win or cover. Features are home-minus-away and are oriented to the chosen team before comparing.',
      'cautions':['Univariate retrospective ranking; many features checked creates multiple-testing risk.',
          '2019-2025 features are from walk-forward computation, but strategy has been researched against these same outcomes.',
          'Missing an ATS bet means failing to cover, not necessarily losing outright.',
          'Correlations are not causal, and no model improvement is claimed until a genuinely untouched evaluation.']}
    return report,stats,d

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True)
    p.add_argument('--selected-bets',required=True)
    p.add_argument('--output',default='results/miss_football_features')
    a=p.parse_args()
    report,stats,detail=audit(pd.read_csv(a.predictions),pd.read_csv(a.selected_bets))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    stats.to_csv(out/'signal_comparison.csv',index=False)
    detail.to_csv(out/'misses_and_wins.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
