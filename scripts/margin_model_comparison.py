"""Walk-forward point-margin forecasting and six-bet spread vs moneyline study.

All margin model fits use strictly earlier weeks/seasons. Predictions are
not derived from actual winning margins of evaluated games.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from nfl_oracle.cli import get_data
from nfl_oracle.core import make_features,eligible,FEATURES
from scripts.consensus_allocator import returns
from scripts.weighted_slate import simulate_weighted

def margin_replay(schedules,first=2019,last=2025):
    game=schedules[schedules.game_type.eq('REG')].copy()
    feat=eligible(make_features(game),min_history=0)
    attrs=game[['game_id','week','spread_line','home_spread_odds','away_spread_odds','home_score','away_score']].drop_duplicates('game_id')
    tab=feat.merge(attrs,on='game_id',validate='one_to_one',suffixes=('','_schedule'))
    # Derived scoring margin is an outcome ONLY for training rows, never a feature.
    tab['margin']=tab.home_score-tab.away_score
    # Use pregame form, not sportsbook point-spread itself, in margin fitting.
    cols=[x for x in FEATURES if x not in ('market_logit',)]
    cols += ['spread_line']
    out=[]
    for year in range(first,last+1):
        for week in sorted(tab.loc[tab.season.eq(year),'week'].unique()):
            train=tab.loc[((tab.season<year)|((tab.season==year)&(tab.week<week)))&tab.margin.notna()]
            test=tab.loc[(tab.season==year)&(tab.week==week)&tab.margin.notna()]
            if len(train)<200 or test.empty:continue
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.))
            model.fit(train[cols],train.margin)
            d=test[['game_id','season','week','home_score','away_score',
                    'spread_line','home_spread_odds','away_spread_odds']].copy()
            d['margin_prediction']=model.predict(test[cols])
            d['true_margin']=test.margin
            out.append(d)
    if not out:raise ValueError('No games for margin evaluation')
    return pd.concat(out,ignore_index=True)

def compare(margin_predictions,winner_predictions,budget=100.):
    _,ml_picks,_=simulate_weighted(winner_predictions,budget=budget,first=int(margin_predictions.season.min()))
    base=ml_picks.loc[ml_picks.stake>0,['game_id','side_home','stake','profit']].rename(columns={'profit':'moneyline_profit'})
    all_data=margin_predictions.merge(base,on='game_id',how='left',validate='one_to_one')
    all_data['ats_edge']=all_data.margin_prediction-all_data.spread_line
    all_data['ats_side_home']=all_data.ats_edge>=0
    all_data['ats_confidence']=abs(all_data.ats_edge)
    # Winner selections remain exactly their original stakes; compare ATS on same side.
    subset=all_data.loc[all_data.stake.notna()].copy()
    subset['ats_side_odds']=np.where(subset.side_home,subset.home_spread_odds,subset.away_spread_odds)
    margin=subset.true_margin-subset.spread_line
    cover=np.where(subset.side_home,margin,-margin)
    payout=returns(subset.ats_side_odds.to_numpy(float))
    subset['same_side_ats_profit']=np.where(np.isclose(cover,0),0,np.where(cover>0,subset.stake*payout,-subset.stake))
    # Separate selection: top-six predicted margin edges, weighted by a smooth signal,
    # full $100 across selected picks, not retrospectively chosen winners.
    slates=[]
    for _,g in all_data.groupby(['season','week']):
        h=g.dropna(subset=['home_spread_odds','away_spread_odds','spread_line']).copy()
        if h.empty:continue
        h=h.sort_values(['ats_confidence','game_id'],ascending=[False,True]).head(6)
        w=np.maximum(.25,h.ats_confidence.to_numpy(float))
        weights=w/w.sum()
        cents=int(round(100*budget))
        stakes=np.floor(cents*weights).astype(int)
        leftover=cents-stakes.sum()
        stakes[np.argsort(-(cents*weights-stakes))[:leftover]]+=1
        h['ats_stake']=stakes/100.
        chosen=np.where(h.ats_side_home,h.home_spread_odds,h.away_spread_odds)
        cover=np.where(h.ats_side_home,h.true_margin-h.spread_line,h.spread_line-h.true_margin)
        pay=returns(chosen.astype(float))
        h['ats_profit']=np.where(np.isclose(cover,0),0,np.where(cover>0,h.ats_stake*pay,-h.ats_stake))
        h['cover']=np.sign(cover)
        slates.append(h)
    ats=pd.concat(slates,ignore_index=True)
    def summarize(d,stake,profit):
        return dict(bets=len(d),stake=round(float(d[stake].sum()),2),
           net=round(float(d[profit].sum()),2),roi=float(d[profit].sum()/d[stake].sum()))
    result={'moneyline_same_games':summarize(subset,'stake','moneyline_profit'),
      'moneyline_picks_bet_as_spread':summarize(subset,'stake','same_side_ats_profit'),
      'margin_model_spread_picks':summarize(ats,'ats_stake','ats_profit'),
      'margin_mae':float(np.abs(margin_predictions.margin_prediction-margin_predictions.true_margin).mean()),
      'market_spread_mae':float(np.abs(margin_predictions.spread_line-margin_predictions.true_margin).mean()),
      'ats_covers':int((ats.cover>0).sum()),'ats_losses':int((ats.cover<0).sum()),
      'ats_pushes':int((ats.cover==0).sum()),
      'warning':'Historically recorded lines and prices may not be executable pregame. Winner data model and margin model share market data. This is a retrospective simulation, not an independently validated wagering edge.'}
    return result,ats,subset

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--winner-predictions',required=True)
    p.add_argument('--output',default='results/margin_model_comparison')
    a=p.parse_args()
    games=get_data(None,list(range(2013,2026)))
    margins=margin_replay(games)
    report,ats,ml=compare(margins,pd.read_csv(a.winner_predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    margins.to_csv(out/'margin_predictions.csv',index=False)
    ats.to_csv(out/'spread_picks.csv',index=False)
    ml.to_csv(out/'paired_moneyline_picks.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
