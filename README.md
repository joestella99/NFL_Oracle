# V6 confidence-first parlay review

V6 adds a confidence-first view for small-edge parlays alongside the existing
V5 value view, with liked-pick audits and per-leg contribution comparisons.
See [the strategy and ten-season diagnostic](docs/v6_conviction_strategy.md).

# V5 cross-market parlays — ten complete seasons (2016–2025)

[`scripts/parlay_v5_ten_year.py`](scripts/parlay_v5_ten_year.py) runs the **official V5 policy unchanged** on all 2016–2025 regular seasons, with an explicit **2014–2015 held-out margin residual warmup**. Every year's residual distribution uses only prior seasons, all ML winner model variants train on previous seasons' out-of-sample predictions, and spread margin predictions are refitted before each week using past games. It evaluates ML+ML, ML+spread, and spread+spread tickets with 2–4 legs; historical spread pushes void their individual leg. Games cannot appear twice in one ticket.

Runs in [GitHub Actions](https://github.com/joestella99/NFL_Oracle/actions) as part of the regular evaluation workflow, outputs under `results/parlay_v5_ten_year/` and the `nfl-oracle-ten-year-replay` artifact. Outputs: `report.json`, `annual.csv`, `weekly.csv`, `by_parlay_length_and_market.csv`, `all_qualifying_tickets.csv`, `top20_each_week_including_rejected.csv`, and `all_single_market_audits.csv`. For every forecast type the report displays full qualification counts and an **equal one-unit per ticket hypothetical ROI**, and **highest conservative-EV eligible ticket per week** for a more concentrated comparison. There is no actual bankroll, wager size, or forced selection; no risk-sizing plan is implied.

Command after generating the existing ten-year forecasts and 2019–2025 margin replay:

`PYTHONPATH=src:. python scripts/parlay_v5_ten_year.py --predictions results/ten_year_comparison/predictions.csv --warmup results/ten_year_comparison/warmup_predictions_2014_2015.csv --recent-margins results/margin_model_comparison/margin_predictions.csv`

**Limits:** Historical moneylines and spreads are not verified same-book as-of quotes; "ROI" assumes the hypothetical compounded prices are tradable and each ticket receives exactly one notional unit. Tickets share game exposures, so their wins and losses are *strongly correlated*. Ten seasons were already consulted repeatedly to build the models; this is retrospective research, not a previously untouched validation set. Only future frozen, time-stamped decisions can establish prospective performance.

---

# NFL Oracle V5 — ML + spread + mixed parlays (percentage-only)

**Current cross-market parlay selection engine:** [`scripts/parlay_mixed_market_v5.py`](scripts/parlay_mixed_market_v5.py), configured in [`config/parlay_mixed_market_policy.json`](config/parlay_mixed_market_policy.json). This expands the official percentage-only 2026 review beyond moneyline-only tickets. Existing V1–V4 research replay scripts remain available; neither sportsbook actions nor bankroll/stake allocation are performed.

**What it does:** For each scheduled NFL game, audit both moneylines and both spread sides (four potential single-leg markets). For tractability the combinatorial search promotes the model-favored ML and best expected-value spread side from each game; generate all cross-game 2-, 3-, and 4-leg combinations of ML+ML, ML+ATS, ATS+ATS (and additional mixed combinations). Rejected combinations remain in the audit. A spread win can occur when the team loses outright. Margin-model spread probabilities come from strictly earlier-season out-of-sample score-margin residuals, with discrete points and explicit ATS push probability. A pushed spread leg refunds/removes only its own multiplier; another losing leg still loses the whole parlay. No two picks from the same game are combined.

**Research criteria:** $0 funding or bankroll inputs; minimum 60% win probability for the model-favored ML side, 42% spread-cover probability for the chosen ATS side, minimum 15% conservative all-leg-win probability, minimum **+2% conservative combined expected value**, and 1% conservative gross price stress per leg. All candidate legs/tickets report probabilities and selection reasons. *These thresholds have not been shown to provide a betting edge.*

**2026 completed game audit:**

`PYTHONPATH=src:. python scripts/parlay_mixed_market_v5.py --forecasts results/parlay_2026_percentages/model_probabilities.csv --prior-margins results/margin_model_comparison/margin_predictions.csv`

**Upcoming schedule / Sunday preview** (explicitly NOT the completed-game historical replay): [`scripts/parlay_live_mixed.py`](scripts/parlay_live_mixed.py) requires either a timestamped same-book ML+ATS quotes CSV (`game_id,bookmaker,captured_at,home_moneyline,away_moneyline,spread_line,home_spread_odds,away_spread_odds`; `spread_line` is home expected winning margin) or a live Odds API key `ODDS_API_KEY` with `--live-odds`:

`PYTHONPATH=src:. python scripts/parlay_live_mixed.py --as-of 2026-10-16T15:38:00-04:00 --quotes-csv my_verified_prices.csv --prior-margins results/margin_model_comparison/margin_predictions.csv`

Or swap `--quotes-csv ...` for `--live-odds` with the key set. No historical 2026 result is used as a future outcome; unfinished games are labeled unknown. The reference forecast uses historical form and market rather than the full historical injury/PBP ensemble; **do not conflate these models**. Quote data and live bookmaker parlay acceptance can differ; no executable tickets or guaranteed positive returns are implied.

---

# 2026 personnel-aware NFL Oracle: percentage-only parlay selection

**Active percent-only selection:** `scripts/parlay_2026_review.py` reads [`config/parlay_percentage_policy.json`](config/parlay_percentage_policy.json). This carries forward the approved **+2% conservative combined-ticket EV / 1% payout deterioration per leg** standard, with no individual-leg EV veto and **NO bankroll, stake or dollar optimization**. Every 2..N cross-game combination is evaluated from estimated probabilities, uncertainty and historical pricing; qualifying is NOT a recommendation to place a real wager.

The script regenerates separately held-out, completed **2026** NFL win forecasts using strictly preceding games, ingests `nflreadpy.load_injuries([2023,2024,2025,2026])` when available, checks source timestamps before kickoff, merges QB and offensive-line out/questionable differences and missingness flags, and refits regularized football probability models with/without injuries on strictly earlier seasons. It chooses among unmodified market, football-only and football+injuries using **2023–2025 Brier only**, *before* scoring 2026; reports 2026 Brier/accuracy for all, plus ticket pass/reject funnel. Weekly 2026 includes only **finished non-tied games**; no future games receive an outcome or retrospective price. It exports `results/parlay_2026_percentages` as a GitHub Actions artifact.

`scripts/capture_espn_current.py` now captures current ESPN NFL injuries and depth-chart endpoints, with *collection timestamps*, via the third-party [Public ESPN endpoint reference](https://github.com/pseudo-r/Public-ESPN-API); this **cannot** be used to backfill past starting quarterbacks or 2023–2025 snapshots. Source is unofficial and may fail/rate limit. Future trust requires an independently recorded confirmed starting-QB source observed before kickoff. Observed contemporary depth charts are projections, not proof of actual starters.

The old 2016–2025 dollar-return V3 simulation and frozen v1 historical baseline remain unchanged for reproducibility. **The percentage-only V4 is not an automated betting feature.** Any historical ROI is a price proxy; it does not prove a future betting edge.

---

# Official NFL Oracle parlay policy: V3 (selected October 16, 2026)

**Default parlay selection entrypoint:** `scripts/parlay_portfolio_v3.py`, using versioned thresholds in [`config/parlay_policy.json`](config/parlay_policy.json). This replaces the old V2 research selector as the preferred **parlay policy**, but does not alter the frozen historical V1 checkpoint, existing straight bet policy, or place actual wagers.

**Official ticket rules:** generate all 2..N cross-game moneyline combinations of favorites with individual predicted win probability >=60%; conservative combined probability >=15%; **conservative ticket EV >=+2%** using **1% gross payout haircut per leg**; *no mandatory positive-EV filter on every individual leg* (weak legs remain explicitly audited). Model probability discount still subtracts a baseline 2 percentage points per leg plus forecast-disagreement penalty.

**Stake rules (hypothetical $1,000 bankroll):** weekly parlay risk <=$50, single ticket <=$5, sum of parlay tickets containing any game <=$20, and straight+parlay tickets containing any game <=$120. Allow multiple parlays and shared teams, never force an eligible ticket merely to increase volume. The straight budget is tracked for overlapping game risk, not deducted from the separate parlay sleeve. No unpriced or unverified wager should be presented as executable.

**Reproduction:** `PYTHONPATH=src:. python scripts/parlay_portfolio_v3.py --predictions results/ten_year_comparison/predictions.csv --warmup results/ten_year_comparison/warmup_predictions_2014_2015.csv --straight-bets results/ten_year_ml_spread_parlays/straight_bets.csv --old-parlays results/ten_year_ml_spread_parlays/calculated_cross_game_parlays.csv`

**Caveat:** The selected model was tuned after inspecting historical 2016–2025 simulations; displayed historical returns are not a clean holdout or evidence of a repeatable real-money edge. Source quotes cannot establish synchronized executable odds. V3 is the official **selection research standard**, not a verified profitable wagering system. V2 stays available for comparison.

---

# NFL Oracle (research prototype)

A small, reproducible NFL moneyline experiment. **It is not a proven betting model.** It does **not** place wagers, guarantee returns, or silently invent odds. The goal is to test whether pregame team form adds predictive information beyond the market's own implied probabilities.

## Frozen checkpoint: NFL Oracle research v1 (October 16, 2026)

**Canonical historical comparator:** `risk_tolerant` in
`scripts/simulate_ten_year_portfolio.py`, with `scripts/checkpoint_v1.py`
saving the full selected straight and parlay tickets, weekly and annual
performance, and a machine-readable manifest in `results/checkpoint_v1/`.

This checkpoint was selected for its **retrospective** 2016–2025 combined
ROI among the strategies compared. It allocates **$100 per eligible week** to
moneyline/spread bets across as many distinct games as clear its risk-adjusted
threshold; if nothing qualifies, it uses a labeled forced fallback moneyline.
It independently researches **at most one optional two-leg cross-game
moneyline parlay** with an **extra $2–$15** experimental stake, priced by
compounding each historical leg's American odds. This parlay selector can
admit some negative estimated expected-value combinations—its historical
return **must not** be mistaken for demonstrated edge.

The larger round-robin experiment, the strict positive-EV parlay selector,
and the broad-spread-capital allocator remain available as **research**, not
silent replacements. The standalone live `nfl-oracle shortlist` uses a
different, simpler predictive model and is **not** equivalent to this
ten-year retrospective portfolio. No automated bets are placed.

To reproduce the checkpoint, run GitHub Actions workflow
`.github/workflows/evaluate.yml` or produce the historical portfolio CSVs
and then:

```sh
PYTHONPATH=src:. python scripts/checkpoint_v1.py \
  --source results/ten_year_ml_spread_parlays --output results/checkpoint_v1
```

**Not a proven profitable model.** Historical quotes are not confirmed
simultaneously executable, the years have been inspected during development,
and positive historical ROI does not guarantee future returns.

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

The research shortlist also produces `experimental_parlay_watchlist.csv`,
`parlay_candidate_audit.csv`, and `parlay_research_report.json`. This
**experimental** two-leg *cross-game moneyline* parlay view uses timestamped
quotes from the **same bookmaker**, excludes games already selected as straight
candidates, and rejects quote snapshots more than 60 minutes apart. It
calculates the normal parlay payout from the two American prices by multiplying
their decimal equivalents. The extra-stake research amount ranges from
$2 to $15 only when its *estimated* return is positive; negative-estimated-
return combinations are labeled **watch only** with $0 suggested.

The expanded research parlay engine evaluates **all cross-game 2-to-N combinations**,
without a maximum number of legs or tickets, and allows overlapping legs
across round-robin tickets and overlap with straight candidates. Candidates
must use one bookmaker and quote snapshots no more than 60 minutes apart.
Each opportunity gets its own estimated EV and a conservative probability
adjustment. Only conservatively positive-EV opportunities qualify for a
nonzero research stake. All qualifying tickets share one adaptive exposure
sleeve, sized as a fraction of a separate configurable bankroll:

```sh
nfl-oracle shortlist --as-of 2026-10-18T11:00 --odds-csv my_quotes.csv --parlay-bankroll 1000
```

An unusually strong slate can increase aggregate parlay exposure, rather than
being arbitrarily capped at $15. The bankroll itself remains a **risk limit**;
it is not automatically spent. Many combinations of overpriced favorites can
all remain watch-only. Overlapping parlays are highly correlated, and the
product of independent leg probabilities is only an approximation.

This is not a verified profitable strategy or a validated live moneyline
model; live shortlist probabilities use the CLI's simpler rolling-form model,
not the richer historical combined-portfolio predictor. Historical calculated
returns cannot prove a particular sportsbook would have accepted both leg
prices simultaneously. Same-game parlays are excluded; no bookmaker-specific
correlated price is invented, and the experiment never places bets.

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

Run `python scripts/replay_weeks.py --season 2026 --weeks 1 2 3 4 5`. The GitHub Actions workflow also runs it and uploads `results/weeks_1_5_2026/` as an artifact containing `report.json`, `predictions.csv`, `weekly_summary.csv`, `confidence_buckets.csv`, and `incorrect_picks_with_features.csv`.

**Average confidence** is the mean probability the model assigned to *its predicted winner* (max of the home and away probability). It is not realized accuracy: compare these side-by-side. The report includes weekly and overall accuracy, sportsbook-favorite accuracy, Brier score, confidence on correct vs wrong picks, confidence ranges, and excluded ties/missing results/missing odds. The primary five-week replay now **refits before every week's games** on earlier seasons and all finished prior weeks of 2026. Early-season games are retained using baseline team-form defaults. Training weights are: **2026 = 4×**, **2025 = 1.5×**, earlier seasons = 1× per game. Weights are experimental, not tuned to these weeks. The output compares this walk-forward model against a frozen pre-2026 model on the same games. `incorrect_picks_with_features.csv` preserves all incorrect predictions and their model inputs for later error analysis. Historical bookmaker moneylines lack verified pregame timestamps, so this is not a tradable-profit claim.


## Enhanced historical replay: QB efficiency and EPA

Run `python scripts/replay_weeks.py --with-pbp --output results/weeks_1_5_2026_advanced` to add prior-game rolling offense/defense EPA per play, matchup differential, the previous game's primary-passer EPA per pass, and passer share. The regular baseline without these features is retained in the same GitHub Actions run for an apples-to-apples comparison.

The walk-forward model refits **before each week**, using all completed earlier weeks in 2026 at **4x sample weight**, 2025 at 1.5x, and earlier seasons at 1x. Games on the same calendar date cannot contribute to one another's features. Historical leading passer is **not a confirmed starting-QB forecast**; this model still lacks injuries and lineup verification. Recorded nflverse sportsbook prices lack verified pregame timestamps.

The initial 2026 Weeks 1–5 advanced test found 50 correct of 78 games (64.1%), Brier 0.231685 compared with basic model 0.231755 and sportsbook 0.228903. These are small-sample findings, **not evidence of profitable betting**. See the Actions evaluation artifact for detailed error files.


## Experimental parlay engine v2 (all eligible favorites, no arbitrary ticket limit)

`python scripts/parlay_portfolio_v2.py --predictions results/ten_year_comparison/predictions.csv --warmup results/ten_year_comparison/warmup_predictions_2014_2015.csv --straight-bets results/ten_year_ml_spread_parlays/straight_bets.csv --old-parlays results/ten_year_ml_spread_parlays/calculated_cross_game_parlays.csv`

Separate **football-specific** probability estimator refits a regularized logistic combiner on **earlier-season out-of-sample model forecasts**. Inputs include three existing models, market anchor, QB efficiency, offense/defense EPA, pass/run matchups, and offensive sustainability. It **does not include** verified starting-QB status, point-in-time injuries or other unavailable data. Its out-of-season Brier score is compared directly to the no-vig market baseline. No labels from the target season are used to fit that season.

For each week: one model-favored ML side per game with estimated win probability >= 60%, **regardless of whether it is also a straight bet**; enumerate every 2..N combination (four eligible games give 6 pairs, 4 triples, 1 four-leg ticket). Audit every ticket. A heuristic, nonstatistical haircut reflects model-vs-market and ensemble disagreement. Additional separate stress subtracts 2% from gross payout **per leg**. Ticket qualifies only when each leg has conservatively positive price-adjusted EV, joint risk-adjusted win probability >= 15%, and ticket conservative EV >= 2%. Those are *research risk parameters*, not proven optimal thresholds. **No wager is forced**, even with four highly probable favorites.

All funded tickets share a $1,000 **research bankroll**: at most 5% weekly total parlay exposure, 0.5% to any one ticket, and 2% exposure to each game **across all overlapping parlays and the pre-existing straight stakes**. Any preexisting straight risk above a cap prevents adding parlay risk to that game; it is not silently reallocated or falsely presented as independent. Ticket count and leg count have no hard maximum, but available budget may fund fewer than the audited possibilities. Outputs: `results/parlay_portfolio_v2/{report.json,weekly.csv,annual.csv,all_combinations_and_exposure.csv}`. Existing historical checkpoint and live shortlist are deliberately unchanged. This is NOT a live booking engine or an established profitable system. Historical moneylines cannot verify executable same-book simultaneous parlay prices; the dataset has been heavily researched and is not a clean holdout.

