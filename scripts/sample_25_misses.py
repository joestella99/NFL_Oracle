"""Deterministic, reproducible 25-miss sample from recent NFL Oracle mixed picks.

Sampling is stratified: 10 losing moneylines and 15 failed ATS tickets,
2023-2025 only, using a fixed RNG seed. No push counts as a miss.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from nfl_oracle.advanced import ADVANCED_FEATURES
from nfl_oracle.matchups import MATCHUP_FEATURES
from nfl_oracle.stability import STABILITY_FEATURES

FEATURES=ADVANCED_FEATURES+MATCHUP_FEATURES+STABILITY_FEATURES

def make_sample(predictions,selected):
    base=predictions.loc[predictions.model.eq('baseline'),['game_id','home_score','away_score','spread_line']+FEATURES]
    picks=selected.loc[selected.strategy.eq('mixed')].copy()
    picks=picks.loc[picks.season.between(2023,2025)&(~picks.won.astype(bool))&(~picks.push.astype(bool))]
    picks=picks.merge(base,on='game_id',how='left',validate='one_to_one')
    output=[]
    for kind,count in [('ML',10),('ATS',15)]:
        group=picks.loc[picks.kind.eq(kind)].sort_values(['season','week','game_id'])
        if len(group)<count:raise ValueError(f'Insufficient {kind} misses: {len(group)}')
        output.append(group.sample(n=count,random_state=20261016))
    d=pd.concat(output,ignore_index=True).sort_values(['season','week','game_id']).reset_index(drop=True)
    d.insert(0,'sample_id',np.arange(1,len(d)+1))
    d['home_margin']=d.home_score-d.away_score
    d['actual_winning_team_side']=np.where(d.home_margin>0,'home','away')
    d['opposing_side']=np.where(d.side.eq('home'),'away','home')
    d['winning_side']=np.where(d.kind.eq('ML'),d.actual_winning_team_side,d.opposing_side)
    d['cover_margin_for_selected']=np.where(d.side.eq('home'),d.home_margin-d.spread_line,d.spread_line-d.home_margin)
    d['selected_side_factor']=np.where(d.side.eq('home'),1.,-1.)
    for feature in FEATURES:
        d['selected_'+feature]=d[feature]*d.selected_side_factor
    assert len(d)==25 and (d.kind.eq('ML').sum()==10) and (d.kind.eq('ATS').sum()==15)
    summary={'sample_size':25,'moneyline_misses':10,'spread_misses':15,
       'eligible_period':'2023-2025','selection':'Deterministic simple random sample within market, 20261016 seed.',
       'total_eligible_misses':int(len(picks)),
       'moneyline_eligible':int(picks.kind.eq('ML').sum()),
       'spread_eligible':int(picks.kind.eq('ATS').sum()),
       'columns':'Positive raw features favor home, negative away. selected_ features orient the pregame advantage to the team Oracle backed; negative selected_ value means the opponent was advantaged.'}
    return summary,d

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--selected-bets',required=True)
    parser.add_argument('--output',default='results/sample_25_misses')
    args=parser.parse_args()
    summary,detail=make_sample(pd.read_csv(args.predictions),pd.read_csv(args.selected_bets))
    path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    detail.to_csv(path/'misses_25_full.csv',index=False)
    selected=['sample_id','season','week','game_id','kind','side','winning_side','home_score','away_score',
        'spread_line','odds','probability','estimated_ev','stake','profit',
        'selected_run_attack_edge','selected_run_attack_vs_defense','selected_run_defense_edge',
        'selected_epa_trimmed_diff','selected_success_rate_diff','selected_explosive_rate_diff',
        'selected_negative_play_diff','selected_epa_std_diff','selected_qb_pass_epa_diff']
    detail[selected].to_csv(path/'misses_25_readable.csv',index=False)
    (path/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
