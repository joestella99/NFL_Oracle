"""Chronologically replay NFL regular-season weeks and audit winner confidence.

Default: weeks 1-5 of 2026. Refit BEFORE each week using prior seasons
and all completed previous weeks in the evaluation season. Give current-season
training games extra weight. Freeze each week before observing its outcomes.
Recorded nflverse odds do NOT establish timestamped executable prices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nfl_oracle.cli import get_data
from nfl_oracle.core import FEATURES, make_features, eligible, fit_model, evaluate
from nfl_oracle.advanced import ADVANCED_FEATURES, make_pregame_advanced
from nfl_oracle.matchups import MATCHUP_FEATURES, matchup_features


DEFAULT_BINS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]


def _number(value: float) -> float | None:
    return round(float(value), 6) if pd.notna(value) and np.isfinite(value) else None


def _summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"games": 0, "correct": 0, "accuracy": None, "average_confidence": None,
                "confidence_minus_accuracy": None, "mean_confidence_correct": None,
                "mean_confidence_wrong": None, "market_accuracy": None,
                "market_average_confidence": None, "brier": None, "market_brier": None,
                "static_accuracy": None, "static_brier": None, "static_average_confidence": None}
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
        "static_accuracy": (_number(((rows["static_home_prob"] >= 0.5).astype(int)
                                    == rows["home_win"].astype(int)).mean())
                            if "static_home_prob" in rows else None),
        "static_brier": (_number(evaluate(rows, "static_home_prob")["brier"])
                         if "static_home_prob" in rows else None),
        "static_average_confidence": (_number(np.maximum(rows["static_home_prob"],
                                                        1 - rows["static_home_prob"]).mean())
                                      if "static_home_prob" in rows else None),
    }


def walk_forward_predictions(games: pd.DataFrame, year: int, weeks: list[int],
                             current_season_weight: float = 4.0,
                             recent_season_weight: float = 1.5,
                             pbp: pd.DataFrame | None = None,
                             include_market: bool = True,
                             include_matchups: bool = False) -> pd.DataFrame:
    """Fit before every week. Never train on the evaluated week or future weeks.

    Historical weeks 1-3 have usable neutral-prior form features even when a
    team has played fewer than 3 games. Using min_history=0 lets the model
    truly learn from *all* previous weeks, including Weeks 1-3 before Week 5.
    This is an explicitly different sample definition from the older replay.
    """
    if current_season_weight < 1 or not np.isfinite(current_season_weight):
        raise ValueError('current_season_weight must be finite and >= 1')
    if recent_season_weight < 1 or not np.isfinite(recent_season_weight):
        raise ValueError('recent_season_weight must be finite and >= 1')
    required = {'game_id', 'season', 'week', 'game_type'}
    if not required.issubset(games):
        raise ValueError(f'Missing week/season identifiers: {sorted(required - set(games.columns))}')
    if games['game_id'].duplicated().any():
        raise ValueError('Schedule must have unique game_ids')
    full = games.loc[games['game_type'].eq('REG')].copy()
    metadata = full[['game_id', 'week']].copy()
    metadata['week'] = pd.to_numeric(metadata['week'], errors='raise').astype(int)
    # Form is always calculated before a game outcome is appended.  The fit
    # itself uses STRICTLY preceding weeks, regardless of available future scores.
    available = eligible(make_features(full), min_history=0).merge(
        metadata, on='game_id', validate='one_to_one', how='left')
    feature_cols = list(FEATURES) if include_market else [x for x in FEATURES if x != "market_logit"]
    if pbp is not None:
        advanced = make_pregame_advanced(full, pbp)
        available = available.merge(advanced, on="game_id", validate="one_to_one", how="left")
        feature_cols += ADVANCED_FEATURES
        available[ADVANCED_FEATURES] = available[ADVANCED_FEATURES].fillna(0.0)
    if include_matchups:
        if pbp is None:
            raise ValueError('Matchup features require play-by-play data')
        matchups=matchup_features(full,pbp)
        available=available.merge(matchups,on='game_id',validate='one_to_one',how='left')
        feature_cols+=MATCHUP_FEATURES
        available[MATCHUP_FEATURES]=available[MATCHUP_FEATURES].fillna(0.)
    old = available.loc[available['season'] < year].copy()
    frozen = fit_model(old, feature_cols=feature_cols)  # Same training sample / feature schema, no current-season updates.
    batches = []
    for week in sorted(set(int(w) for w in weeks)):
        training = available.loc[(available['season'] < year) |
                                 ((available['season'] == year) & (available['week'] < week))].copy()
        test = available.loc[(available['season'] == year) & (available['week'] == week)].copy()
        if test.empty:
            continue
        # Any ties and unfinished games were excluded by eligible().  Crucially,
        # week N observations cannot enter this training window, even when
        # historical data already contains their final results.
        weights = np.ones(len(training), dtype=float)
        weights[training['season'].to_numpy() == year - 1] = recent_season_weight
        weights[training['season'].to_numpy() == year] = current_season_weight
        model = fit_model(training, sample_weight=weights, feature_cols=feature_cols)
        test['model_home_prob'] = model.predict_proba(test[feature_cols])[:, 1]
        test['static_home_prob'] = frozen.predict_proba(test[feature_cols])[:, 1]
        test['training_games'] = len(training)
        test['uses_advanced_pbp'] = pbp is not None
        test['uses_market_feature'] = include_market
        test['current_season_training_games'] = int((training['season'] == year).sum())
        test['training_through_week'] = week - 1
        test['current_season_weight'] = current_season_weight
        batches.append(test)
    if not batches:
        raise ValueError(f'No completed eligible games for {year} weeks {weeks}')
    return pd.concat(batches, ignore_index=True)


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
    if "static_home_prob" in eligible and not eligible["static_home_prob"].between(0, 1).all():
        raise ValueError("Invalid probability in static_home_prob")
    eligible = eligible.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)

    weekly = []
    for w in weeks:
        sched_week = chosen.loc[pd.to_numeric(chosen["week"]) == w]
        done_week = sched_week.loc[sched_week["is_complete"]]
        ties = int(done_week["is_tie"].sum())
        not_scored = int((~sched_week["is_complete"]).sum())
        no_model = int((done_week["_prediction_join"].eq("left_only") & ~done_week["is_tie"]).sum())
        sample = eligible.loc[eligible["week"] == w]
        row = {"week": w, "scheduled": int(len(sched_week)), "ties_excluded": ties,
               "unscored_excluded": not_scored, "no_prediction_excluded": no_model,
               **_summary(sample)}
        if "training_games" in sample and len(sample):
            row.update({"training_games": int(sample["training_games"].iloc[0]),
                        "current_season_training_games": int(sample["current_season_training_games"].iloc[0]),
                        "training_through_week": int(sample["training_through_week"].iloc[0])})
        weekly.append(row)

    bins = []
    for low, high in DEFAULT_BINS:
        b = eligible.loc[(eligible["confidence"] >= low) & (eligible["confidence"] < high)]
        bins.append({"range": f"{low:.0%}-{min(high, 1.0):.0%}", **_summary(b)})

    report = {
        "season": year,
        "weeks": weeks,
        "training_policy": (f"Walk-forward; refit each week with all prior seasons and "
                            f"current-season completed weeks before the predicted week. "
                            f"Current season weight {float(eligible['current_season_weight'].iloc[0]):g}x; "
                            f"previous season 1.5x; older seasons 1x."
                            if "current_season_weight" in eligible else "Precomputed model probabilities"),
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
            "4x weighting was a hypothesis, not a cross-validated or optimized parameter.",
            "Static comparison uses the same zero-minimum-form sample for an apples-to-apples evaluation.",
        ],
    }
    columns = ["week", "gameday", "game_id", "away_team", "home_team",
               "predicted_winner", "actual_winner", "correct", "confidence",
               "model_home_prob", "market_home_prob", "market_correct", "market_confidence",
               "home_moneyline", "away_moneyline", "home_score", "away_score",
               "static_home_prob", "training_games", "current_season_training_games",
               "training_through_week", "current_season_weight", "uses_advanced_pbp", *FEATURES, *ADVANCED_FEATURES]
    columns = list(dict.fromkeys([c for c in columns if c in eligible.columns]))
    return report, eligible[columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL Oracle multiweek confidence simulation")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--weeks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-start", type=int, default=2020)
    parser.add_argument("--current-season-weight", type=float, default=4.0,
                        help="Weight for earlier completed games from the evaluated season")
    parser.add_argument("--schedule-csv", help="Optional local nflverse-format schedule CSV")
    parser.add_argument("--market-blind", action="store_true", help="Exclude sportsbook probability from model features")
    parser.add_argument("--matchups", action="store_true", help="Include pass/run scheme and opponent-relative matchup features")
    parser.add_argument("--with-pbp", action="store_true", help="Fetch nflverse PBP and add prior-game EPA/QB efficiency inputs")
    parser.add_argument("--pbp-csv", help="Use a cached play-by-play CSV instead of downloading")
    parser.add_argument("--output", default="results/weeks_1_5_2026")
    args = parser.parse_args()
    if min(args.weeks) < 1 or max(args.weeks) > 18:
        parser.error("Weeks must be between 1 and 18")
    if args.train_start >= args.season:
        parser.error("--train-start must precede --season")
    games = get_data(args.schedule_csv, list(range(args.train_start, args.season + 1)))
    pbp = pd.read_csv(args.pbp_csv) if args.pbp_csv else None
    if (args.with_pbp or args.matchups) and pbp is None:
        import nflreadpy as nfl
        pbp = nfl.load_pbp(list(range(args.train_start, args.season + 1))).select(
            ["game_id", "posteam", "defteam", "epa", "play_type", "passer_player_id"]
        ).to_pandas()
    predicted = walk_forward_predictions(games, year=args.season, weeks=args.weeks,
                                         pbp=pbp,
                                         current_season_weight=args.current_season_weight,
                                         include_market=not args.market_blind,
                                         include_matchups=args.matchups)
    report, table = build_report(predicted, games, args.season, args.weeks)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "predictions.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(report["by_week"]).to_csv(out / "weekly_summary.csv", index=False)
    pd.DataFrame(report["by_confidence_bucket"]).to_csv(out / "confidence_buckets.csv", index=False)
    table.loc[~table["correct"]].to_csv(out / "incorrect_picks_with_features.csv", index=False)
    print(json.dumps(report, indent=2))
    print("\nIndividual picks:")
    print(table[["week", "away_team", "home_team", "predicted_winner", "actual_winner",
                 "confidence", "correct"]].to_string(index=False))
    print(f"\nSaved reproducible results to {out.resolve()}")


if __name__ == "__main__":
    main()