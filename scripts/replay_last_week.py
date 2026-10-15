"""Replay October 8-12, 2026 using only results available before each game.

Recorded nflverse moneylines are not proven pregame timestamped quotes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nfl_oracle.cli import get_data
from nfl_oracle.core import backtest, evaluate

START = pd.Timestamp("2026-10-08")
END = pd.Timestamp("2026-10-13")  # inclusive through Monday Oct 12
OUT = Path("results/week5_2026")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seasons = list(range(2020, 2027))
    games = get_data(None, seasons)
    report, predicted = backtest(games, test_year=2026)
    dates = pd.to_datetime(predicted["gameday"])
    week = predicted.loc[(dates >= START) & (dates < END)].copy()
    if week.empty:
        raise RuntimeError(
            "No scored NFL games with recorded odds for Oct 8-12 2026 in nflverse. "
            "Cannot claim week-level prediction accuracy."
        )
    outcomes = games[["game_id", "home_score", "away_score"]].drop_duplicates("game_id")
    week = week.merge(outcomes, on="game_id", how="left", validate="one_to_one")
    week["actual_winner"] = np.where(week["home_win"] == 1, week["home_team"], week["away_team"])
    week["predicted_winner"] = np.where(week["model_home_prob"] >= 0.5, week["home_team"], week["away_team"])
    week["market_winner"] = np.where(week["market_home_prob"] >= 0.5, week["home_team"], week["away_team"])
    week["correct"] = week["predicted_winner"] == week["actual_winner"]
    week["market_correct"] = week["market_winner"] == week["actual_winner"]
    result = {
        "evaluation_dates": "2026-10-08 through 2026-10-12",
        "completed_games_in_window_with_usable_odds": int(len(week)),
        "excluded": "Games with tied scores, insufficient form history, or invalid/missing moneylines are excluded.",
        "model": evaluate(week),
        "market": evaluate(week, "market_home_prob"),
        "correct_model_picks": int(week["correct"].sum()),
        "correct_market_favorites": int(week["market_correct"].sum()),
        "train_seasons": "2020-2025",
        "test_season": "2026",
        "limitations": "Recorded bookmaker lines lack verified pregame timestamps. No gambling profitability claim."
    }
    week.sort_values(["gameday", "game_id"]).to_csv(OUT/"predictions.csv", index=False)
    (OUT/"report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(week[["gameday", "away_team", "home_team", "away_score", "home_score",
                "predicted_winner", "actual_winner", "correct", "model_home_prob"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
