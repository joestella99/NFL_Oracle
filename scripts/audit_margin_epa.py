"""Point-in-time scoring-margin versus offensive-EPA disagreement research.

Uses historical pregame predictions. Fit a constrained market+signal logistic
correction on one season, evaluate on a strictly later season. A model that does
not beat the sportsbook is NOT promoted. This evaluates all eligible games,
not only the examples that happen to support the hypothesis.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss

EPS=1e-6

def tag_groups(df):
    d=df.copy()
    required={'market_home_prob','home_win','form_margin_diff','epa_offense_diff','game_id'}
    missing=required-set(d)
    if missing:raise ValueError(f'Missing pregame variables: {sorted(missing)}')
    if d.game_id.duplicated().any():raise ValueError('Duplicate game IDs')
    if d[list(required-{'game_id'})].isna().any().any():raise ValueError('Missing outcomes or input features')
    margin=d.form_margin_diff.to_numpy(float)
    epa=d.epa_offense_diff.to_numpy(float)
    # Fixed research bands (zero does not suggest either team).
    d['group']=np.select([(margin>0)&(epa>0),(margin<0)&(epa<0),
        (margin<0)&(epa>0),(margin>0)&(epa<0)],
        ['both_home','both_away','hidden_home_epa','hidden_away_epa'],default='neutral')
    d['epa_signal_home']=epa>0
    d['margin_signal_home']=margin>0
    d['epa_better_than_margin']=np.sign(epa)!=np.sign(margin)
    d['home_market_surprise']=d.home_win-d.market_home_prob
    d['market_brier_individual']=(d.home_win-d.market_home_prob)**2
    d['epa_correct']=(epa>0)==(d.home_win==1)
    d['margin_correct']=(margin>0)==(d.home_win==1)
    return d

def by_group(df):
    out=[]
    for group,g in df.groupby('group',sort=True):
        out.append(dict(group=group,n=len(g),home_win_rate=float(g.home_win.mean()),
            market_home_probability=float(g.market_home_prob.mean()),
            market_home_residual=float(g.home_market_surprise.mean()),
            brier=float(g.market_brier_individual.mean()),
            epa_accuracy=float(g.epa_correct.mean()),
            margin_accuracy=float(g.margin_correct.mean())))
    return out

def correction_features(d):
    market=np.clip(d.market_home_prob.to_numpy(float),EPS,1-EPS)
    logit=np.log(market/(1-market))
    margin=np.clip(d.form_margin_diff.to_numpy(float)/14,-3,3)
    epa=np.clip(d.epa_offense_diff.to_numpy(float)/0.25,-3,3)
    # Distinct non-collinear information: EPA-minus-margin, and interaction.
    return np.column_stack((logit,epa-margin,np.sign(margin)*np.sign(epa),epa*margin))

def evaluate(train,test):
    if train.season.max()>=test.season.min():
        raise ValueError('Training seasons must be strictly before evaluation season')
    train=tag_groups(train)
    test=tag_groups(test)
    x=correction_features(train);y=train.home_win.to_numpy(int)
    model=LogisticRegression(C=0.05,max_iter=1000)
    model.fit(x,y)
    scores=test.copy()
    scores['corrected_home_prob']=model.predict_proba(correction_features(test))[:,1]
    scores['corrected_pick_home']=scores.corrected_home_prob>=.5
    scores['market_pick_home']=scores.market_home_prob>=.5
    summary={}
    for name,col in [('market','market_home_prob'),('margin_epa_correction','corrected_home_prob')]:
        prob=scores[col].to_numpy(float)
        summary[name]=dict(brier=float(brier_score_loss(scores.home_win,prob)),
            log_loss=float(log_loss(scores.home_win,prob,labels=[0,1])),
            accuracy=float(((prob>=.5)==scores.home_win).mean()))
    return dict(train_games=len(train),test_games=len(scores),
        train_seasons=sorted(int(x) for x in train.season.unique()),
        test_seasons=sorted(int(x) for x in test.season.unique()),
        groups=by_group(scores),all_games=summary,
        improvement_brier=summary['market']['brier']-summary['margin_epa_correction']['brier'],
        promotion='research_only',
        notes=['Model correction fit on earlier season only; no threshold tuned on test season.',
               'Simple EPA-versus-margin group analysis is descriptive, not a validated edge.',
               'Sportsbook odds in nflverse have no verified pregame timestamp.']),scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True,help='Multi-season decisions.csv export including the pregame features')
    p.add_argument('--train-season',type=int,default=2025)
    p.add_argument('--test-season',type=int,default=2026)
    p.add_argument('--output',default='results/margin_epa_audit')
    a=p.parse_args()
    d=pd.read_csv(a.predictions)
    if 'season' not in d:
        if 'test_season' not in d:raise ValueError('Season column required')
        d['season']=d.test_season
    t=d[d.season==a.train_season].copy()
    e=d[d.season==a.test_season].copy()
    if t.empty or e.empty:raise ValueError('Need both complete training and evaluation seasons')
    result,scores=evaluate(t,e)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(result['groups']).to_csv(out/'groups.csv',index=False)
    scores.to_csv(out/'game_audit.csv',index=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
