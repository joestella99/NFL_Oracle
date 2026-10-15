"""Chronological out-of-sample calibration on ten-year replay predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

def design(df):
    m=np.clip(df.market_home_prob.to_numpy(float),1e-5,1-1e-5)
    p=np.clip(df.prediction.to_numpy(float),1e-5,1-1e-5)
    return np.column_stack((logit(m),logit(p)-logit(m)))

def rolling_calibration(frame,first_test=2019,min_history=2,C=.05):
    keys={'game_id','season','model','home_win','market_home_prob','prediction'}
    if not keys.issubset(frame):raise ValueError(f'Missing {keys-set(frame)}')
    if frame.duplicated(['game_id','model']).any():raise ValueError('Duplicate game/model')
    if min_history<1:raise ValueError('min_history must be >=1')
    predictions=[];summary=[]
    for name,g in frame.groupby('model'):
        for season in sorted(g.season.unique()):
            train=g[g.season<season]
            test=g[g.season==season].copy()
            if season<first_test or train.season.nunique()<min_history:continue
            model=LogisticRegression(C=C,max_iter=1000).fit(design(train),train.home_win.astype(int))
            test['calibrated_prob']=model.predict_proba(design(test))[:,1]
            test['last_calibration_season']=int(train.season.max())
            predictions.append(test)
            for label,col in [('market','market_home_prob'),('raw','prediction'),('calibrated','calibrated_prob')]:
                p=test[col].to_numpy(float); y=test.home_win.to_numpy(int)
                summary.append({'season':int(season),'model':name,'predictor':label,
                    'games':len(y),'correct':int(((p>=.5)==y).sum()),
                    'accuracy':float(((p>=.5)==y).mean()),
                    'brier':float(np.mean((p-y)**2))})
    if not predictions:raise ValueError('No eligible evaluation years')
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(summary)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--first-test',type=int,default=2019)
    parser.add_argument('--output',default='results/rolling_calibration')
    a=parser.parse_args()
    d,s=rolling_calibration(pd.read_csv(a.predictions),first_test=a.first_test)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/'predictions.csv',index=False);s.to_csv(out/'season_summary.csv',index=False)
    total=[]
    for name,g in d.groupby('model'):
        for key,col in [('market','market_home_prob'),('raw','prediction'),('calibrated','calibrated_prob')]:
            p=g[col].to_numpy(float);y=g.home_win.to_numpy(int)
            total.append({'model':name,'predictor':key,'games':len(g),'correct':int(((p>=.5)==y).sum()),
                          'accuracy':float(((p>=.5)==y).mean()),'brier':float(np.mean((p-y)**2))})
    report={'overall':total,'policy':'Fit every season on strictly earlier seasons out-of-sample predictions, C=.05 fixed in advance; evaluate 2019-2025.', 'warning':'Retrospective validation; feature families were designed after viewing 2026 outcomes.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
