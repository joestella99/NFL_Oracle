"""Out-of-sample injury feature ablation, 2023-24 fit, 2025 test.

Uses NFLverse injury reports with pre-kickoff last-modified cutoff.
These reports do not establish that an injured QB was a starting QB.
Missing reports are explicitly flagged, not assumed to mean healthy.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from nfl_oracle.cli import get_data
from scripts.export_injury_reports import extract

def injury_features(snapshots,games):
    if snapshots.empty:raise ValueError('No injury reports')
    records=snapshots.copy()
    records['status_known']=records.status_known.astype(int)
    wide=records.pivot(index='game_id',columns=['team','role'],values='out_count') if False else None
    home=games[['game_id','home_team','away_team']].copy()
    rows=[]
    for g in home.itertuples(index=False):
        r={'game_id':g.game_id}
        for role in ('qb','ol'):
            vals=[]
            for team in (g.home_team,g.away_team):
                sub=records[(records.game_id==g.game_id)&(records.team==team)&(records.role==role)]
                if len(sub)==0 or not bool(sub.status_known.iloc[0]):
                    vals.append(None)
                else:vals.append(float(sub.out_count.iloc[0]))
            r[role+'_home_out']=vals[0]
            r[role+'_away_out']=vals[1]
            r[role+'_coverage']=int(vals[0] is not None and vals[1] is not None)
            r[role+'_out_diff']=(vals[1]-vals[0]) if r[role+'_coverage'] else 0.
        rows.append(r)
    return pd.DataFrame(rows)

def experiment(predict,features):
    p=predict[predict.model.eq('baseline')].merge(features,on='game_id',validate='one_to_one')
    train=p[p.season.isin([2023,2024])].copy()
    test=p[p.season.eq(2025)].copy()
    if train.empty or test.empty:raise ValueError('Insufficient historical observations')
    for d in (train,test):
        m=np.clip(d.market_home_prob.to_numpy(float),.001,.999)
        d['logit_market']=np.log(m/(1-m))
    base_cols=['logit_market']
    injury_cols=base_cols+['qb_out_diff','ol_out_diff','qb_coverage','ol_coverage']
    # The injury model has the same baseline feature and train/test games.
    results=[]
    for name,cols in [('market_only_fitted',base_cols),('market_plus_injuries',injury_cols)]:
        model=LogisticRegression(C=.05,max_iter=2000)
        model.fit(train[cols],train.home_win.astype(int))
        prob=model.predict_proba(test[cols])[:,1]
        result={'model':name,'games':len(test),'brier':float(brier_score_loss(test.home_win,prob)),
                'accuracy':float(((prob>=.5)==test.home_win).mean())}
        results.append(result)
    market=test.market_home_prob.to_numpy(float)
    results.insert(0,{'model':'sportsbook','games':len(test),
        'brier':float(brier_score_loss(test.home_win,market)),
        'accuracy':float(((market>=.5)==test.home_win).mean())})
    report={'fit_seasons':[2023,2024],'test_season':2025,'results':results,
        'test_games_with_both_qb_reports':int(test.qb_coverage.sum()),
        'test_games_with_both_ol_reports':int(test.ol_coverage.sum()),
        'limitations':'Out counts reflect reported injured players, not starters. Missing status is encoded with coverage indicators. Reporting may be incomplete or reflect revisions; no performance gain is presumed.'}
    return report,test

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='results/injury_ablation')
    args=p.parse_args()
    import nflreadpy as nfl
    years=[2023,2024,2025]
    schedules=get_data(None,years)
    injuries=nfl.load_injuries(years).to_pandas()
    snaps,audit=extract(schedules,injuries)
    feats=injury_features(snaps,schedules)
    report,test=experiment(pd.read_csv(args.predictions),feats)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    test.to_csv(out/'test_features.csv',index=False)
    audit.to_csv(out/'source_coverage.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
