"""Confidence bucket audit on the same ten-year held-out games.

Without timestamped historical odds, this is a hit-rate/calibration audit,
NOT a realizable betting ROI simulation. Bookmaker no-vig probabilities
are insufficient to compute actual expected profits.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def confidence_audit(predictions):
    rows=[]
    for name,g in [('market',predictions[predictions.model=='baseline'])]+[
        (model,predictions[predictions.model==model]) for model in sorted(predictions.model.unique())]:
        p=g.market_home_prob.to_numpy(float) if name=='market' else g.prediction.to_numpy(float)
        y=g.home_win.to_numpy(int)
        c=np.maximum(p,1-p)
        correct=(p>=.5)==y
        if {'home_moneyline','away_moneyline'}.issubset(g):
            odds=np.where(p>=.5,g.home_moneyline.to_numpy(float),g.away_moneyline.to_numpy(float))
            profits=np.where(correct,np.where(odds>0,odds/100,100/(-odds)),-1.)
        else:
            profits=np.full(len(p),np.nan)
        for low,hi in ((.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.00001)):
            idx=(c>=low)&(c<hi)
            n=int(idx.sum());wins=int(correct[idx].sum())
            rows.append({'predictor':name,'bucket':f'{int(100*low)}-{min(100,int(100*hi))}%',
                'games':n,'wins':wins,'losses':n-wins,'win_rate':wins/n if n else None,
                'average_confidence':float(c[idx].mean()) if n else None,
                'fair_decimal_odds_at_realized_rate':n/wins if wins else None,
                'flat_stake_roi_proxy':float(profits[idx].mean()) if n and np.isfinite(profits[idx]).all() else None,
                'flat_stake_profit_units':float(profits[idx].sum()) if n and np.isfinite(profits[idx]).all() else None})
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',default='results/high_confidence')
    a=parser.parse_args()
    result=confidence_audit(pd.read_csv(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'buckets.csv',index=False)
    report={'rows':result.where(pd.notna(result),None).to_dict('records'),
       'limitations':'For an actual positive-EV bet, offered odds must exceed fair odds based on defensibly estimated forward probability. Recorded historical moneylines may be closing or retrospectively revised, are not independently verified as executable at prediction time, and returns omit fees/limits.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(result[result.bucket=='80-90%'].to_string(index=False))
if __name__=='__main__':main()
