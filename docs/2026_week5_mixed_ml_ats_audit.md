# 2026 Weeks 1–5: mixed moneyline and point-spread parlay research (V5)

**Run:** https://github.com/joestella99/NFL_Oracle/actions/runs/38958283399. All three jobs passed; **101 unit tests succeeded** after handling unknown future outcomes safely. Official engine `scripts/parlay_mixed_market_v5.py` with thresholds in `config/parlay_mixed_market_policy.json`. Prospective Sunday engine `scripts/parlay_live_mixed.py` requires both ML and spread timestamped quotes **from the same bookmaker** via quote CSV or `ODDS_API_KEY`; no live market shortlist has yet been run from an authenticated quote provider.

## Summary

For **2026 Week 5** (14 completed eligible non-tied games), each game is reviewed for both sides of the moneyline and both sides of the spread; one favored moneyline plus the highest-EV ATS side is advanced from each game for pair/triple/four-leg combinations, with no same-game mixing. Spread pushes reduce/void that leg and are not treated as automatic parlay losses. There are no stake, dollar or bankroll allocation gates.

| Forecast variant | Week 5 candidates | Week 5 qualifying (+2% conservative EV, 15% joint all-win) | All completed 2026 qualifying |
| --- | ---: | ---: | ---: |
| Market no-vig winner probabilities + empirical margin ATS | 8,827 | **0** | **0** |
| Football winner probability + empirical margin ATS | 10,431 | **0** | **11** |
| Football+injuries winner probability + empirical margin ATS | 10,431 | **0** | **10** |

The **market-calibrated baseline is still preferred** because football+injury and football-only versions underperformed market probability accuracy in 2023–25 and completed 2026 evaluation. The 11 experimental football-model qualifying tickets consist of 1 in Week 2, 8 in Week 3 and 2 in Week 4. **Only 1 of 11 experimental selections had a positive equal-unit settlement** and 10 lost. These combinations share games and are correlated; the outcome is not a valid independent-trial profit or predictive-validation claim. Do not adopt the less accurate football variant just because it generated more candidates.

### Breakdown, market-calibrated Week 5

- 246 ML+ML-only combinations
- 7,125 combinations mixing moneylines and spreads
- 1,456 spread+spread-only combinations
- 7,067 rejected first on less than 15% conservative all-win chance
- Remaining 1,760 failed the +2% conservative combined ticket EV screen
- **0 positive +2% conservative-EV tickets**, despite the larger candidate universe

Top rejected cross-game two-spread ticket: Green Bay + Chicago matchup **Green Bay-side ATS** plus Jets + Browns matchup **Cleveland-side ATS** (NFLverse game ID `2026_05_CHI_GB:away:ATS | 2026_05_CLE_NYJ:away:ATS`); estimated model EV **+11.11% before** uncertainty and 1%/leg payout stress, **-2.75% after**. Both historical legs lost. The conservative combined probability of both winning was **27.18%**. This illustrates the importance of explicitly reporting raw and stressed EV: a positive unadjusted prediction isn't a trustworthy wager if its probability model is not established.

Week 5 initial calculated probability for Browns-side ATS (away in `2026_05_CLE_NYJ`) was **54.85%** with raw leg EV **+7.09%**, which fell to **-0.14%** under conservative risk adjustments and ultimately failed to cover. Market favorite MLs have a negative unadjusted EV based on their *own* no-vig sportsbook probabilities, by construction, because the actual payout includes the vig. Thus moving from ML-only to ML/ATS is necessary to consider more markets but cannot magically establish systematic profitable prices.

### What this fixes and what it cannot prove

- Fixed: 2026 past-weeks parlay construction now explicitly evaluates spread and moneyline legs across different games; no arbitrary one-parlay limit; every selection and exclusion is auditable; ATS pushes are properly treated; no stake sizing decisions; future-game as-of preview route exists for verified full-market quotes.
- Not yet fixed: historical NFLverse pricing still lacks synchronized real-world execution timestamps; the margin residual distribution is only a proxy and not a validated 2026 ATS edge; the market Brier still beats the football win models. Live Sunday pair forecasts need time-stamped same-book ML+spread quotes (The Odds API `ODDS_API_KEY` or supplied CSV) for the entire Sunday slate. Do **not** mark an upcoming pair as definitively rejected solely because it wasn't included in a completed-games replay.
- This backtest keeps the selected conservative EV >= 2%, min conservative all-win >= 15%, price haircut 1% per leg, with ML leg >=60% and ATS leg >=42% probability. These are research settings, not guaranteed profitable bet rules.

Artifacts are attached to the successful run at `parlay_mixed_market_2026/{report.json,2026_weekly_market_comparison.csv,all_four_game_market_sides.csv,mixed_parlay_ticket_audit.csv,top_twenty_even_if_rejected.csv}`.
