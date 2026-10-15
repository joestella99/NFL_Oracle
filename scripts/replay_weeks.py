"""Chronologically replay NFL regular-season weeks and audit winner confidence.

Default: weeks 1-5 of 2026. The model is trained on preceding seasons,
while prior games *within* the evaluation season may inform team form.
Recorded nflverse odds do NOT establish timestamped executable prices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nfl_oracle.cli import get_data
from nfl_oracle.core import backtest, evaluate


DEFAULT_BINS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]


def _number(value: float) -> float | None:
    return round(float(value), 6) if pd.notna(value) and np.isfinite(value) else None


def _summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"games": 0, "correct": 0, "accuracy": None, "average_confidence": None,
                "confidence_minus_accuracy": None, "mean_confidence_correct": None,
                "mean_confidence_wrong": None, "market_accuracy": None,
                "market_average_confidence": None, "brier": None, "market_brier": None}
    return {
        "games": int(len(rows)),
        "correct": int(rows["correct"].sum()),
        "accuracy": _number(rows["correct"].mean()),
        "average_confidence": _number(rows["confidence"].mean()),
        "confidence_minus_accuracy": _number(rows["confidence"].mean() - rows["correct"].mean()),
        "mean_confidence_correct": _number(rows.loc[rows["correct"], "confidence"].mean()),
        "mean_confidence_wrong": _number(rows.loc[~rows["correct"], "confidence"].mean()),
        "market_accuracy": _number(rows["market_correct"].mean()),
        "market_average_confidence": _number(rows["market_confidence"].mean()),
        "brier": _number(evaluate(rows)["brier"]),
        "market_brier": _number(evaluate(rows, "market_home_prob")["brier"]),
    }


def build_report(predictions: pd.DataFrame, schedules: pd.DataFrame,
                 year: int, weeks: list[int]) -> tuple[dict, pd.DataFrame]:
    """Summarize precomputed held-out probabilities; never fill missing games.

    Ties, unfinished games, missing pregame form or missing moneylines are
    explicitly counted as exclusions; zero eligible games is an error.
    """
    required = {"game_id", "season", "week", "game_type", "home_score", "away_score"}
    missing = required.difference(schedules.columns)
    if missing:
        raise ValueError(f"Schedule missing columns: {sorted(missing)}")
    weeks = sorted(set(int(w) for w in weeks))
    selected = schedules.loc[
        (pd.to_numeric(schedules["season"], errors="coerce") == year)
        & schedules["game_type"].eq("REG")
        & pd.to_numeric(schedules["week"], errors="coerce").isin(weeks)
    ].copy()
    if selected.empty:
        raise ValueError(f"No regular-season games for {year}, weeks {weeks}")
    if selected["game_id"].duplicated().any():
        raise ValueError("Duplicate game_ids in the selected schedule")
    score_h = pd.to_numeric(selected["home_score"], errors="coerce")
    score_a = pd.to_numeric(selected["away_score"], errors="coerce")
    selected["is_complete"] = score_h.notna() & score_a.notna()
    selected["is_tie"] = selected["is_complete"] & score_h.eq(score_a)

    if predictions["game_id"].duplicated().any():
        raise ValueError("Duplicate game_ids in model predictions")
    slim = predictions.drop(columns=["season", "week", "game_type"], errors="ignore")
    chosen = selected.merge(slim, on="game_id", how="left", suffixes=("_schedule", ""),
                            validate="one_to_one", indicator="_prediction_join")
    has_prediction = chosen["_prediction_join"].eq("both")
    eligible = chosen.loc[chosen["is_complete"] & ~chosen["is_tie"] & has_prediction].copy()
    if eligible.empty:
        raise ValueError("No completed, non-tie games with eligible held-out predictions")
    if not eligible[["model_home_prob", "market_home_prob", "home_win"]].notna().all().all():
        raise ValueError("Predicted probabilities and actual outcomes cannot be missing")
    for col in ["model_home_prob", "market_home_prob"]:
        if not eligible[col].between(0, 1).all():
            raise ValueError(f"Invalid probability in {col}")

    # Model chooses home for exactly 50%; confidence is P(chosen side wins).
    eligible["predicted_home"] = eligible["model_home_prob"] >= 0.5
    eligible["confidence"] = np.where(eligible["predicted_home"],
                                      eligible["model_home_prob"], 1 - eligible["model_home_prob"])
    eligible["correct"] = eligible["predicted_home"].astype(int).eq(eligible["home_win"].astype(int))
    eligible["predicted_winner"] = np.where(eligible["predicted_home"], eligible["home_team"], eligible["away_team"])
    eligible["actual_winner"] = np.where(eligible["home_win"].eq(1), eligible["home_team"], eligible["away_team"])
    eligible["market_predicted_home"] = eligible["market_home_prob"] >= 0.5
    eligible["market_confidence"] = np.where(eligible["market_predicted_home"],
                                             eligible["market_home_prob"], 1 - eligible["market_home_prob"])
    eligible["market_correct"] = eligible["market_predicted_home"].astype(int).eq(eligible["home_win"].astype(int))
    eligible["week"] = pd.to_numeric(eligible["week"], errors="raise").astype(int)
    eligible = eligible.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)

    weekly = []
    for w in weeks:
        sched_week = chosen.loc[pd.to_numeric(chosen["week"]) == w]
        done_week = sched_week.loc[sched_week["is_complete"]]
        ties = int(done_week["is_tie"].sum())
        not_scored = int((~sched_week["is_complete"]).sum())
        no_model = int((done_week["_prediction_join"].eq("left_only") & ~done_week["is_tie"]).sum())
        sample = eligible.loc[eligible["week"] == w]
        weekly.append({"week": w, "scheduled": int(len(sched_week)), "ties_excluded": ties,
                       "unscored_excluded": not_scored, "no_prediction_excluded": no_model,
                       **_summary(sample)})

    bins = []
    for low, high in DEFAULT_BINS:
        b = eligible.loc[(eligible["confidence"] >= low) & (eligible["confidence"] < high)]
        bins.append({"range": f"{low:.0%}-{min(high, 1.0):.0%}", **_summary(b)})

    report = {
        "season": year,
        "weeks": weeks,
        "training_seasons": f"< {year}",
        "definition": "Confidence = the model's probability that its selected winner wins, always >=50%.",
        "scope": "Completed, untied, regular-season games with sufficient previous team form and valid historical moneylines.",
        "overall": _summary(eligible),
        "by_week": weekly,
        "by_confidence_bucket": bins,
        "limitations": [
            "nflverse moneylines do not establish time-stamped, tradable pregame odds.",
            "Historical sportsbook probabilities are a model input as well as the comparison baseline.",
            "Confidence is a predicted probability, not validated reliability; compare it with realized accuracy.",
            "A five-week sample is small; probability calibration estimates will be noisy.",
        ],
    }
    columns = ["week", "gameday", "game_id", "away_team", "home_team",
               "predicted_winner", "actual_winner", "correct", "confidence",
               "model_home_prob", "market_home_prob", "market_correct", "market_confidence",
               "home_moneyline", "away_moneyline"]
    return report, eligible[columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL Oracle multiweek confidence simulation")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--weeks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-start", type=int, default=2020)
    parser.add_argument("--schedule-csv", help="Optional local nflverse-format schedule CSV")
    parser.add_argument("--output", default="results/weeks_1_5_2026")
    args = parser.parse_args()
    if min(args.weeks) < 1 or max(args.weeks) > 18:
        parser.error("Weeks must be between 1 and 18")
    if args.train_start >= args.season:
        parser.error("--train-start must precede --season")
    games = get_data(args.schedule_csv, list(range(args.train_start, args.season + 1)))
    _, predicted = backtest(games, test_year=args.season)
    report, table = build_report(predicted, games, args.season, args.weeks)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "predictions.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(report["by_week"]).to_csv(out / "weekly_summary.csv", index=False)
    pd.DataFrame(report["by_confidence_bucket"]).to_csv(out / "confidence_buckets.csv", index=False)
    print(json.dumps(report, indent=2))
    print("\nIndividual picks:")
    print(table[["week", "away_team", "home_team", "predicted_winner", "actual_winner",
                 "confidence", "correct"]].to_string(index=False))
    print(f"\nSaved reproducible results to {out.resolve()}")


if __name__ == "__main__":
    main()