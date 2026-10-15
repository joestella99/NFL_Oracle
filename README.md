# NFL Oracle (research prototype)

A small, reproducible NFL moneyline experiment. **It is not a proven betting model.** It does **not** place wagers, guarantee returns, or silently invent odds. The goal is to test whether pregame team form adds predictive information beyond the market's own implied probabilities.

## What it predicts

- Home-team **win probability** in a regular-season NFL game (not covers/spreads or player props).
- Relative value against recorded American moneylines (`home_moneyline`, `away_moneyline`).
- A candidate shortlist only when modeled probability and expected value exceed configured thresholds; `NO EDGE` is a valid result.

## Data

Public [nflverse schedule data](https://nflreadr.nflverse.com/reference/load_schedules.html), loaded with [nflreadpy](https://nflreadpy.nflverse.com/). Historical rows contain game results, teams, rest days, and lines. You can also pass an offline nflverse-format CSV. Actual betting-price history and pregame timestamps **are not guaranteed** by that dataset; a retrospective return calculation using its lines is only a rough proxy, not a verified tradable backtest. Access to time-stamped bookmaker odds is necessary for serious deployment.

Current features: normalized market win probability, each team's prior-eight-games rolling points for, points against, scoring margin, win rate and total points, plus pregame rest difference. The model uses regularized logistic regression and compares against the no-vig market baseline.

All team form is computed **before** appending the result of that game. The classifier is trained only on seasons **prior** to the held-out season; no random shuffling across seasons. We do **not** claim a real predictive advantage until a historical backtest proves it.

## Install

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Run a real backtest

The included GitHub Actions workflow can also run this automatically after this project is pushed to its own repository. It downloads real historical data and uploads the held-out report as an artifact; **no real-data score has been verified in this development environment.**


```sh
# Downloads NFL schedules if internet is available
nfl-oracle backtest --seasons 2020 2021 2022 2023 2024 2025 --test-year 2025

# Or: reuse a predownloaded nflverse schedule CSV
nfl-oracle backtest --schedule-csv data/games.csv --test-year 2025
```

Outputs `results/backtest_report.json` and `results/backtest_predictions.csv` containing out-of-sample predictions, game IDs and actual game outcomes, plus a bookmaker-line proxy ROI for **simulated flat $1 bets** passing a 5-percentage-point edge and 5%-EV rule. No real money is wagered.

## Generate a research watchlist for a Sunday

Provide an independently recorded quote file (see `examples/odds_template.csv`): columns `game_id,home_moneyline,away_moneyline,bookmaker,captured_at`. `captured_at` is an ISO-8601 timestamp with a timezone, e.g. `2026-10-18T14:00:00Z`. The program rejects quotes newer than the supplied as-of timestamp or older than 12 hours.

```sh
nfl-oracle shortlist --as-of 2026-10-18T11:00 --odds-csv my_quotes.csv

# Or: pull currently available moneylines from The Odds API (requires a valid key/tier).
export ODDS_API_KEY='your-private-key-here'
nfl-oracle shortlist --as-of 2026-10-18T11:00 --live-odds
```

The output is named `research_watchlist.csv`, **not betting advice**. The live option writes `live_quotes_audit.csv` to preserve exact quote provenance. Live odds access and pricing depend on the odds provider's current account tier. It is not validated to beat the sportsbook. Future results from the as-of date onward are excluded even if a retrospective schedule data file includes them. No odds means no picks.

## Hard limitations

1. The small feature set is inadequate for claiming genuine NFL edges. It lacks starting-QB status, injuries, EPA/play, opponent-adjusted strength, weather, and player availability.
2. Market inputs may be recorded closing lines; the historical data source doesn't prove what exact odds were available at a particular decision time. Need timestamped bookmaker quotes before claiming historical executable profits.
3. A profitable small backtest can be luck. Validate on multiple untouched seasons and compare probability quality with the market before selecting a staking policy.
4. Predicting NFL game winners is highly competitive; professional sportsbooks incorporate most obvious signal already. There may be **no bet** that passes a credible edge test on a Sunday.
5. This initial version uses team scoring form, not the comparable-player matchup algorithm from Bat Oracle. Player reception and rushing props are a later experimental module after collecting historical prop odds.

## Three-Sunday $20 challenge

Sundays: Oct 18, Oct 25, Nov 1, 2026. Turning $20 into $100 is a **5x** result. Three all-in wins at approximately -140 odds would achieve about $100.75, but all-in staking risks losing the entire $20 and the implied chance of three independent fair-price wins is approximately 20%. We recommend treating $20 as a strict maximum entertainment loss, never chasing and never asserting guaranteed profits. Software should be evaluated on probability calibration, Brier score, and out-of-sample ROI rather than whether one high-variance three-week streak hits.

## Privacy, source usage and data notes

No API key is required for public nflverse downloads. Never check API credentials into GitHub. Verify source attribution and terms if later monetizing the tool.

## Replay last week's games (Oct 8-12, 2026)

Run `python scripts/replay_last_week.py` to predict the completed 2026 Week 5 games from 2020-2025 training, using only earlier results for team form. Prints the per-game winners and correctness, and saves `results/week5_2026/report.json` and `predictions.csv`. Tied games and games missing usable lines are excluded. GitHub Actions runs this replay automatically on push. These evaluations use historical lines without verified timestamps; they are not evidence of executable betting edge.

## Multiweek confidence audit: 2026 Weeks 1–5

Run `python scripts/replay_weeks.py --season 2026 --weeks 1 2 3 4 5`. The GitHub Actions workflow also runs it and uploads `results/weeks_1_5_2026/` as an artifact containing `report.json`, `predictions.csv`, `weekly_summary.csv`, and `confidence_buckets.csv`.

**Average confidence** is the mean probability the model assigned to *its predicted winner* (max of the home and away probability). It is not realized accuracy: compare these side-by-side. The report includes weekly and overall accuracy, sportsbook-favorite accuracy, Brier score, confidence on correct vs wrong picks, confidence ranges, and excluded ties/missing results/missing odds. Training uses only seasons before 2026; rolling team features use only previous games. Historical bookmaker moneylines lack verified pregame timestamps, so this is an experiment, not a tradable-profit claim.
