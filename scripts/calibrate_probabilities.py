"""Fit a logistic second-stage calibrator on an earlier held-out season; evaluate on later games."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

def inputs(df):
    market=np.clip(df["market_home_prob"].to_numpy(dtype=float),.001,.999)
    model=np.clip(df["model_home_prob"].to_numpy(dtype=float),.001,.999)
    return np.column_stack((logit(market),logit(model)-logit(market)))

def outcomes(df):
    if "home_win" in df:
        return df["home_win"].to_numpy(dtype=int)
    if df[["home_score","away_score"]].isna().any().any() or (df["home_score"]==df["away_score"]).any():
        raise ValueError("Missing scores or tied games")
    return (df["home_score"]>df["away_score"]).to_numpy(dtype=int)

def evaluate(calibration,evaluation,C=1.0, evaluation_season=None):
    if C<=0: raise ValueError("C must be positive")
    if "season" not in evaluation and evaluation_season is not None:
        evaluation=evaluation.copy()
        evaluation["season"]=int(evaluation_season)
    if "season" not in calibration or "season" not in evaluation:
        raise ValueError("Both datasets need a season column, or supply --evaluation-season")
    if calibration["season"].max()>=evaluation["season"].min():
        raise ValueError("Calibration seasons must precede evaluation")
    for d in (calibration,evaluation):
        if d[["market_home_prob","model_home_prob"]].isna().any().any():
            raise ValueError("Missing probabilities")
    y=outcomes(calibration)
    if len(np.unique(y))<2: raise ValueError("Both calibration outcomes required")
    calibrator=LogisticRegression(C=C,max_iter=1000).fit(inputs(calibration),y)
    output=evaluation.copy()
    output["calibrated_home_prob"]=calibrator.predict_proba(inputs(output))[:,1]
    actual=outcomes(output)
    # Conservative directional prediction, separate from the calibrated probability.
    # This rule retains market winner accuracy; it does NOT improve probabilistic
    # Brier score by changing or disguising predicted probabilities.
    output["decision_home_market_guard"]=output["market_home_prob"].to_numpy(dtype=float)>=0.5
    output["decision_home_calibrated"]=output["calibrated_home_prob"].to_numpy(dtype=float)>=0.5
    output["decision_home_original"]=output["model_home_prob"].to_numpy(dtype=float)>=0.5
    metrics={}
    for name,col in [("market","market_home_prob"),("original","model_home_prob"),("calibrated","calibrated_home_prob")]:
        p=output[col].to_numpy(dtype=float)
        metrics[name]={"brier":float(brier_score_loss(actual,p)),
                       "log_loss":float(log_loss(actual,p,labels=[0,1])),
                       "winner_accuracy":float(((p>=.5)==actual).mean())}
    return {"calibration_games":len(calibration),"evaluation_games":len(evaluation),
            "decision_accuracy_market_guard":float((output["decision_home_market_guard"].to_numpy()==actual).mean()),
            "decision_accuracy_calibrated":float((output["decision_home_calibrated"].to_numpy()==actual).mean()),
            "regularization_C":C,"metrics":metrics,
            "caveat":"2025 base and 2026 advanced model use different feature sets; validate matched pipelines on additional seasons."},output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration",required=True)
    p.add_argument("--evaluation",required=True)
    p.add_argument("--output",default="results/calibrated_2026")
    p.add_argument("--C",type=float,default=1.0)
    p.add_argument("--evaluation-season",type=int,default=None)
    a=p.parse_args()
    report,rows=evaluate(pd.read_csv(a.calibration),pd.read_csv(a.evaluation),a.C,a.evaluation_season)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    rows.to_csv(output/"predictions.csv",index=False)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
