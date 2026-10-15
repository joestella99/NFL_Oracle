"""Audit out-of-sample model errors without treating correlations as causal signals.

Usage: python scripts/audit_errors.py --predictions results/weeks_1_5_2026_advanced/predictions.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SIGNALS = ["form_margin_diff", "form_win_pct_diff", "rest_diff",
           "epa_offense_diff", "epa_defense_diff", "epa_matchup_diff",
           "qb_pass_epa_diff", "qb_pass_share_diff"]


def audit(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    needed = {"game_id", "week", "correct", "model_home_prob", "market_home_prob", "home_score", "away_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    df = df.copy()
    for col in ["model_home_prob", "market_home_prob", "home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not df[["model_home_prob", "market_home_prob"]].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Invalid predicted probability")
    if df.game_id.duplicated().any():
        raise ValueError("Duplicate games")
    df["home_win"] = (df.home_score > df.away_score).astype(int)
    df["model_home_pick"] = df.model_home_prob >= .5
    df["market_home_pick"] = df.market_home_prob >= .5
    df["is_correct"] = df.model_home_pick.eq(df.home_win.astype(int))
    df["market_correct"] = df.market_home_pick.eq(df.home_win.astype(int))
    df["confidence"] = np.maximum(df.model_home_prob, 1-df.model_home_prob)
    df["market_confidence"] = np.maximum(df.market_home_prob, 1-df.market_home_prob)
    df["model_market_disagree"] = df.model_home_pick.ne(df.market_home_pick)
    df["market_probability_gap"] = (df.model_home_prob - df.market_home_prob).abs()
    df["is_market_upset"] = ~df.market_correct
    n = len(df)
    if not n:
        raise ValueError("No evaluated games")
    wrong = df.loc[~df.is_correct]
    correct = df.loc[df.is_correct]
    def rate(subset):
        return {"games": int(len(subset)), "accuracy": round(float(subset.is_correct.mean()), 4) if len(subset) else None,
                "average_confidence": round(float(subset.confidence.mean()),4) if len(subset) else None}
    signals = []
    for feature in SIGNALS:
        if feature not in df:
            continue
        absolute = pd.to_numeric(df[feature],errors="coerce").abs()
        a = absolute.loc[~df.is_correct].dropna()
        b = absolute.loc[df.is_correct].dropna()
        signals.append({"feature":feature,"wrong_n":len(a),"correct_n":len(b),
                        "wrong_mean_abs":round(float(a.mean()),5) if len(a) else None,
                        "correct_mean_abs":round(float(b.mean()),5) if len(b) else None,
                        "difference_wrong_minus_correct":round(float(a.mean()-b.mean()),5) if len(a) and len(b) else None})
    # These are descriptive slices, not statistical significance or causal features.
    report = {
        "scope":"Retrospective error diagnosis on already-held-out predictions; not an independently validated new model",
        "games":n, "model":rate(df), "market_accuracy":round(float(df.market_correct.mean()),4),
        "wrong":int(len(wrong)),
        "wrong_picks_also_market_upsets":int(wrong.is_market_upset.sum()),
        "agreement_with_market":rate(df.loc[~df.model_market_disagree]),
        "disagreement_with_market":rate(df.loc[df.model_market_disagree]),
        "mean_confidence_wrong":round(float(wrong.confidence.mean()),4) if len(wrong) else None,
        "mean_confidence_correct":round(float(correct.confidence.mean()),4) if len(correct) else None,
        "mean_model_market_probability_gap":round(float(df.market_probability_gap.mean()),4),
        "weekly": [{"week":int(w),**rate(group)} for w,group in df.groupby("week",sort=True)],
        "signal_slices":signals,
        "next_experiment":"Test market-free and market-aware QB/EPA models on additional held-out seasons using only pregame information",
        "cautions":["Correlations on 78 games are noisy and do not prove feature importance.",
                    "Recorded historical moneylines are not verified kickoff-time quotes.",
                    "Do not tune to these 2026 outcomes and then claim 2026 is an unbiased holdout."]
    }
    return report, df.loc[~df.is_correct].copy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("results/error_audit"))
    args=parser.parse_args()
    report, wrong=audit(pd.read_csv(args.predictions))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    wrong.to_csv(args.output/"incorrect_predictions.csv",index=False)
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()