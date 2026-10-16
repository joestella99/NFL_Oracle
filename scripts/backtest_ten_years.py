"""Full-season, 10-year NFL Oracle model comparison with genuine walk-forward training.

One set of point-in-time feature tables is shared across model variants. Every
season/week uses only strictly prior weeks/years for fitting. No hyperparameter
selection uses the evaluated year's outcomes. Includes regular season only.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from nfl_oracle.cli import get_data
from nfl_oracle.core import make_features,eligible,FEATURES,fit_model
from nfl_oracle.advanced import make_pregame_advanced,ADVANCED_FEATURES
from nfl_oracle.matchups import matchup_features,MATCHUP_FEATURES
from nfl_oracle.stability import make_stability,STABILITY_FEATURES

def compare(games,pbp,first=2016,last=2025):
    if first>=last: raise ValueError("Require at least two test seasons")
    full=games.loc[games.game_type.eq('REG')].copy()
    info=full[['game_id','week','spread_line','home_spread_odds','away_spread_odds','home_score','away_score']].drop_duplicates('game_id')
    rows=eligible(make_features(full),min_history=0).merge(info,on='game_id',validate='one_to_one')
    for fn,cols in [(make_pregame_advanced,ADVANCED_FEATURES),
                    (matchup_features,MATCHUP_FEATURES),(make_stability,STABILITY_FEATURES)]:
        features=fn(full,pbp)
        rows=rows.merge(features,on='game_id',how='left',validate='one_to_one')
        rows[cols]=rows[cols].fillna(0.)
    variants={'baseline':FEATURES,
              'matchup':FEATURES+ADVANCED_FEATURES+MATCHUP_FEATURES,
              'sustainability':FEATURES+ADVANCED_FEATURES+MATCHUP_FEATURES+STABILITY_FEATURES}
    outputs=[]
    for year in range(first,last+1):
        for week in sorted(rows.loc[rows.season.eq(year),'week'].unique()):
            train=rows.loc[(rows.season<year)|((rows.season==year)&(rows.week<week))]
            test=rows.loc[rows.season.eq(year)&rows.week.eq(week)].copy()
            if len(train)<70 or test.empty:continue
            weights=np.where(train.season.eq(year),4.,np.where(train.season.eq(year-1),1.5,1.))
            for name,cols in variants.items():
                model=fit_model(train,sample_weight=weights,feature_cols=cols)
                p=model.predict_proba(test[cols])[:,1]
                outputs.append(pd.DataFrame({'season':year,'week':week,'game_id':test.game_id,
                   'model':name,'home_win':test.home_win.astype(int),
                   'market_home_prob':test.market_home_prob,'prediction':p,
                   'home_moneyline':test.home_moneyline,'away_moneyline':test.away_moneyline,
                   'spread_line':test.spread_line,'home_spread_odds':test.home_spread_odds,
                   'away_spread_odds':test.away_spread_odds,
                   'home_score':test.home_score,'away_score':test.away_score,
                   **{column:test[column] for column in ADVANCED_FEATURES+MATCHUP_FEATURES+STABILITY_FEATURES}}))
    if not outputs:raise ValueError('No test games')
    out=pd.concat(outputs,ignore_index=True)
    if out.duplicated(['game_id','model']).any():raise ValueError('Duplicate predictions')
    n_models=len(variants)
    if not out.groupby('game_id').model.nunique().eq(n_models).all():
        raise ValueError('Model variants must use identical held-out games')
    def stats(df,col):
        return dict(games=len(df),
          correct=int(((df[col]>=.5)==df.home_win).sum()),
          accuracy=float(((df[col]>=.5)==df.home_win).mean()),
          brier=float(np.mean((df[col]-df.home_win)**2)))
    per_season=[]
    for (year,name),sub in out.groupby(['season','model']):
        model_stats=stats(sub,'prediction')
        market_stats=stats(sub,'market_home_prob')
        per_season.append(dict(season=int(year),model=name,**model_stats,
                               market_correct=market_stats['correct'],
                               market_accuracy=market_stats['accuracy'],
                               market_brier=market_stats['brier'],
                               brier_delta_vs_market=model_stats['brier']-market_stats['brier']))
    totals=[]
    for name,sub in out.groupby('model'):
        model_stats=stats(sub,'prediction');market_stats=stats(sub,'market_home_prob')
        totals.append(dict(model=name,**model_stats,market_brier=market_stats['brier'],
          market_accuracy=market_stats['accuracy'],brier_delta_vs_market=model_stats['brier']-market_stats['brier']))
    report=dict(first=first,last=last,train_from=int(rows.season.min()),
      sample_policy='All completed non-tied regular-season games with valid moneylines; minimum history 0.',
      training='Before each week: earlier seasons and earlier weeks only; current season 4x, previous 1.5x.',
      total=totals,per_season=per_season,
      limitations=['NFLverse historical odds have no verified pregame timestamps.',
      'Feature designs and fixed weights were developed during 2026 experimentation; this is retrospective validation, not a virgin holdout.',
      'Winner accuracy alone does not demonstrate positive betting expected value.'])
    return report,out

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--first',type=int,default=2016)
    a.add_argument('--last',type=int,default=2025)
    a.add_argument('--train-from',type=int,default=2013)
    a.add_argument('--output',default='results/ten_year_comparison')
    args=a.parse_args()
    if args.train_from>=args.first:a.error('Training must precede first test season')
    seasons=list(range(args.train_from,args.last+1))
    games=get_data(None,seasons)
    import nflreadpy as nfl
    pbp=nfl.load_pbp(seasons).select(
      ['game_id','posteam','defteam','epa','play_type','passer_player_id']).to_pandas()
    report,table=compare(games,pbp,args.first,args.last)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(report['per_season']).to_csv(output/'season_summary.csv',index=False)
    table.to_csv(output/'predictions.csv',index=False)
    print(json.dumps({'total':report['total'],'per_season':report['per_season']},indent=2))
if __name__=='__main__':main()
