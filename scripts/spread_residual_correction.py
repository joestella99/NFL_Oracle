"""Chronological spread-margin residual correction using pregame football stats.

Train correction to existing margin forecasts on strictly prior observations.
Compare MAE and ATS side accuracy before deployment.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

SIGNALS=['run_attack_edge','run_attack_vs_defense','run_defense_edge',
 'epa_trimmed_diff','success_rate_diff','negative_play_diff','explosive_rate_diff',
 'epa_std_diff','pass_attack_vs_defense']
def evaluate(margins,history):
    features=history.loc[history.model.eq('baseline'),['game_id']+SIGNALS]
    d=margins.merge(features,on='game_id',validate='one_to_one')
    d['residual']=d.true_margin-d.margin_prediction
    out=[]
    for season in sorted(d.season.unique()):
        if season<2021:continue
        train=d.loc[d.season<season]
        test=d.loc[d.season==season].copy()
        if len(train)<300:continue
        estimator=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=500.))
        estimator.fit(train[SIGNALS],train.residual)
        test['corrected_margin']=test.margin_prediction+estimator.predict(test[SIGNALS])
        out.append(test)
    df=pd.concat(out,ignore_index=True)
    def metrics(x):
        actual=x.true_margin.to_numpy(float)
        baseline=x.margin_prediction.to_numpy(float)
        corrected=x.corrected_margin.to_numpy(float)
        target=x.spread_line.to_numpy(float)
        no_push=~np.isclose(actual,target)
        def correct(pred):
            return float(np.mean(((pred-target)>0)[no_push]==((actual-target)>0)[no_push]))
        return {'games':len(x),'baseline_mae':float(mean_absolute_error(actual,baseline)),
             'corrected_mae':float(mean_absolute_error(actual,corrected)),
             'market_mae':float(mean_absolute_error(actual,target)),
             'baseline_ats_side_accuracy_ex_pushes':correct(baseline),
             'corrected_ats_side_accuracy_ex_pushes':correct(corrected)}
    report={'all':metrics(df),'development_2021_2022':metrics(df.loc[df.season<=2022]),
            'later_2023_2025':metrics(df.loc[df.season>=2023]),
            'limitations':['Residual correction is trained on prior seasons only, but features were selected after inspecting historical results.',
              'ATS side accuracy is not ROI and does not account for market vig.',
              'Baseline forecasts themselves use historical spread lines without quote timestamps.']}
    return report,df
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--margin-predictions',required=True)
    p.add_argument('--winner-predictions',required=True)
    p.add_argument('--output',default='results/spread_residual_correction')
    a=p.parse_args()
    report,df=evaluate(pd.read_csv(a.margin_predictions),pd.read_csv(a.winner_predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    df.to_csv(out/'corrected_margins.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
