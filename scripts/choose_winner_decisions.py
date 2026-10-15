"""Select a market-override threshold on an earlier season; evaluate later games unchanged.

Winner accuracy is a separate decision metric from the probability model's Brier score.
Historical nflverse odds are not independently verified tradable snapshots.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS=(0.0,0.01,0.015,0.02,0.03,0.04,0.05,0.06)

def target(df):
    if 'home_win' in df:
        return df['home_win'].to_numpy(dtype=int)
    if not {'home_score','away_score'}.issubset(df):
        raise ValueError('Need scored outcomes or home_win')
    if df[['home_score','away_score']].isna().any().any():
        raise ValueError('Unscored games')
    if (df.home_score==df.away_score).any():
        raise ValueError('Ties must be excluded')
    return (df.home_score>df.away_score).to_numpy(dtype=int)

def decide(df,threshold):
    if threshold<0: raise ValueError('Threshold must be nonnegative')
    market=df['market_home_prob'].to_numpy(dtype=float)
    model=df['model_home_prob'].to_numpy(dtype=float)
    if not np.isfinite(market).all() or not np.isfinite(model).all():
        raise ValueError('Missing probabilities')
    if (market<0).any() or (market>1).any() or (model<0).any() or (model>1).any():
        raise ValueError('Invalid probabilities')
    market_pick=market>=0.5
    model_pick=model>=0.5
    override=(model_pick!=market_pick)&(np.abs(model-market)>=threshold)
    return np.where(override,model_pick,market_pick),override

def select_and_evaluate(train,test,thresholds=THRESHOLDS):
    if 'season' not in train:
        raise ValueError('Calibration season required')
    if 'season' in test and train.season.max()>=test.season.min():
        raise ValueError('Decision-selection data must precede evaluation season')
    y_train=target(train)
    scores=[]
    for t in thresholds:
        decisions,overrides=decide(train,t)
        scores.append({'threshold':t,'accuracy':float((decisions==y_train).mean()),
                       'correct':int((decisions==y_train).sum()),
                       'overrides':int(overrides.sum())})
    # Select by past-season accuracy, favoring fewer overrides when equally accurate.
    best=sorted(scores,key=lambda x:(-x['correct'],x['overrides'],-x['threshold']))[0]
    chosen,changed=decide(test,best['threshold'])
    actual=target(test)
    market=(test.market_home_prob.to_numpy(dtype=float)>=0.5)
    model=(test.model_home_prob.to_numpy(dtype=float)>=0.5)
    detail=test.copy()
    detail['actual_home_win']=actual
    detail['decision_home']=chosen
    detail['market_home']=market
    detail['model_home']=model
    detail['changed_market_pick']=changed
    detail['correct']=chosen==actual
    report={
      'selection_years':sorted(map(int,train.season.unique())),
      'train_games':len(train),'test_games':len(test),
      'selected_threshold':best['threshold'],
      'selection_scores':scores,
      'evaluation':{
        'selected_correct':int((chosen==actual).sum()),
        'selected_accuracy':float((chosen==actual).mean()),
        'market_correct':int((market==actual).sum()),
        'market_accuracy':float((market==actual).mean()),
        'model_correct':int((model==actual).sum()),
        'overrides':int(changed.sum()),
        'correct_overrides':int(((chosen==actual)&changed).sum()),
        'incorrect_overrides':int(((chosen!=actual)&changed).sum())
      },
      'limitations':'2025 and 2026 use different feature pipelines; 2026 sample is 78 games. Decision threshold only: does not improve Brier score, prove betting value, or validate execution prices.'
    }
    return report,detail

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',required=True)
    p.add_argument('--evaluation',required=True)
    p.add_argument('--evaluation-season',type=int,default=None)
    p.add_argument('--output',default='results/winner_decisions_2026')
    args=p.parse_args()
    selection=pd.read_csv(args.selection)
    evaluation=pd.read_csv(args.evaluation)
    if 'season' not in evaluation and args.evaluation_season:
        evaluation['season']=args.evaluation_season
    report,detail=select_and_evaluate(selection,evaluation)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    detail.to_csv(out/'decisions.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
