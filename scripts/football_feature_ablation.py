"""Chronological football-feature upgrades: ablation of rushing and sustained EPA.

Uses historical OUT-OF-WEEK pregame features exported by ten_year_comparison.
Fits only on earlier seasons/weeks, compares market-only, market+run,
market+sustained, and full features. No automatic production promotion.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

EXPERIMENTS={
 'market_only':[],
 'rushing':['run_attack_edge','run_attack_vs_defense','run_defense_edge'],
 'sustainable':['epa_trimmed_diff','success_rate_diff','negative_play_diff','explosive_rate_diff'],
 'combined':['run_attack_edge','run_attack_vs_defense','run_defense_edge','epa_trimmed_diff','success_rate_diff','negative_play_diff','explosive_rate_diff']
}
def evaluate(pred):
    d=pred.loc[pred.model.eq('baseline')].copy()
    if d.duplicated('game_id').any():raise ValueError('Duplicate games')
    d['market_logit']=logit(d.market_home_prob.clip(.0001,.9999))
    rows=[]
    for season in sorted(d.season.unique()):
        if season<2019:continue
        for week in sorted(d.loc[d.season.eq(season),'week'].unique()):
            train=d.loc[(d.season<season)|((d.season==season)&(d.week<week))]
            test=d.loc[d.season.eq(season)&d.week.eq(week)]
            if len(train)<250:continue
            for name,cols in EXPERIMENTS.items():
                features=['market_logit']+cols
                clf=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                    LogisticRegression(C=.04,max_iter=500))
                clf.fit(train[features],train.home_win)
                p=clf.predict_proba(test[features])[:,1]
                rows.extend(dict(game_id=gid,season=int(season),week=int(week),
                  model=name,p=float(prob),home_win=int(y),market=float(m))
                  for gid,prob,y,m in zip(test.game_id,p,test.home_win,test.market_home_prob))
    out=pd.DataFrame(rows)
    def stat(x):
        y=x.home_win.to_numpy(int);p=x.p.to_numpy(float)
        return {'games':int(len(x)),'accuracy':float(((p>=.5)==y).mean()),
                'brier':float(np.mean((p-y)**2)),
                'logloss':float(np.mean(-y*np.log(np.clip(p,1e-8,1))-(1-y)*np.log(np.clip(1-p,1e-8,1))))}
    report={'overall':{},'later_evaluation':{},'development':{}}
    for name,g in out.groupby('model'):
        report['overall'][name]=stat(g)
        for period,mask in [('development',g.season<=2022),('later_evaluation',g.season>=2023)]:
            report[period][name]=stat(g.loc[mask])
    report['warning']='Retrospective feature discovery after analyzing prior outcomes; 2023-25 is NOT a virgin holdout. Compare performance with market-only before promotion.'
    return report,out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='results/football_feature_ablation')
    a=p.parse_args()
    report,rows=evaluate(pd.read_csv(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    rows.to_csv(out/'predictions.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
