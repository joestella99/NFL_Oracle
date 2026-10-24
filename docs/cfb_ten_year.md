# College Football Oracle — ten-season historical research replay

**Execution:** GitHub Actions [CFB 2016–2025 walk-forward replay](https://github.com/joestella99/NFL_Oracle/actions/workflows/cfb_ten_year.yml), successfully completed October 24, 2026. This is a replay, **not** evidence of sustainable profitability or executable historical bets.

## Summary, one-unit stakes per selected ticket (2016–2025)

| Portfolio | Bets | Wins | Losses | Pushes | Units profit | Hypothetical ROI |
|---|---:|---:|---:|---:|---:|---:|
| Single bets, all markets | 1,648 | 622 | 1,007 | 19 | -104.1821 | **-6.32%** |
| Moneyline singles | 786 | 207 | 579 | 0 | -56.6153 | **-7.20%** |
| Spread singles | 862 | 415 | 428 | 19 | -47.5668 | **-5.52%** |
| One 2-leg ticket per qualifying week | 123 | 26 | 97 | 0 | +19.3751 | **+15.75%** |
| One 3-leg ticket per qualifying week | 57 | 9 | 48 | 0 | +7.7879 | **+13.66%** |

*The positive parlay figures are very uncertain, especially because the leg mix includes hypothetical prices and the payouts are arithmetic products of individual leg prices, not verified accepted parlays.*

A simple independent-ticket bootstrap (12,000 replicates, resampling observed returns) gave the following **illustrative**, not formal prospective, 95% percentile intervals:

- 2-leg ROI: **−25.7% to +61.8%**, 65/123 tickets contain one or more assumed -110 ATS prices; maximum cumulative unit drawdown about 21.68.
- 3-leg ROI: **−51.2% to +88.9%**, 45/57 tickets contain one or more assumed -110 ATS prices; maximum cumulative unit drawdown 18. The bootstrap ignores overlapping-game exposures and nonstationarity.

These intervals include large losses, so **do not infer a betting edge from the positive parlay averages**.

## Annual results

| Year | Eligible singles | Singles ROI | ML ROI | ATS ROI | 2-leg ROI | 3-leg ROI |
|---|---:|---:|---:|---:|---:|---:|
| 2016 | 64 | -1.91% | +2.89% | -8.09% | +72.76% | -100% |
| 2017 | 51 | -25.03% | -35.79% | -10.84% | -60.39% | +159.61% |
| 2018 | 67 | -1.01% | -8.29% | +10.48% | +123.41% | +155.48% |
| 2019 | 64 | -4.99% | -17.24% | +12.91% | +14.53% | -100% |
| 2020 | 166 | +3.01% | unavailable | +3.01% | +59.45% | +39.16% |
| 2021 | 227 | -10.50% | -6.27% | -15.69% | -24.43% | -100% |
| 2022 | 253 | -2.26% | +11.62% | -16.93% | +20.68% | -100% |
| 2023 | 246 | -14.31% | -28.05% | -1.43% | -61.82% | +57.93% |
| 2024 | 249 | -10.17% | -15.54% | -4.39% | -41.93% | -39.26% |
| 2025 | 261 | -0.48% | +6.36% | -8.27% | +44.84% | +91.76% |

2020 ML ROI unavailable because no qualifying matched moneyline bets occurred, **not** because ML bets broke even.

## Policy and chronology

Source: [sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data), pinned to Git commit `09f2595e1b23d485691577611d0d1f9cccbb4331`. We used season schedules and an archived sportsbook betting quote dataset. The schedules include CFBD pregame Elo fields. There are 7,160 scored FBS-versus-FBS regular-season games with ratings in the test period. Training begins in 2014, with 2016 predictions based only on 2014–2015 outcomes. Each subsequent season refits using only earlier seasons, never its test-season results.

- Logistic regression on pregame Elo difference and home field estimates ML win probabilities; 65% archived no-vig ML consensus within the chosen book plus 35% model. 
- Ridge regression on pregame Elo difference and home field estimates score margin. 60% archived implied spread margin plus 40% model margin; prior-season residual SD (minimum 15 points) maps to spread cover probability.
- The decision engine considers both market sides but chooses at most one positive-stressed-EV leg per game, with an absolute 2-point probability stress, 1% profit-odds stress, and >=2% conservative EV selection rule.
- For each season/week, one highest-EV 2-leg ticket and one highest-EV 3-leg ticket, each from the six top eligible cross-game legs **at one shared sportsbook**. Different-book combinations are rejected; the probability product still assumes independence and a book-accepted parlay payout is not proven.
- Books are chosen by a *fixed* provider priority, not the retrospectively best quote. No betting stakes, compounding, promotions, limits or actual slips.
- Spread pushes void individual spread legs, and an otherwise losing parlay is still a loser.

## Important caveats

1. **No proof of price availability before kickoff.** Archived odds lack independently verifiable point-in-time quote capture and bookmaker acceptance. A later-migrated quote is not proof of a wagerable line.
2. **ATS pricing is often missing.** Exactly 760 of 862 selected ATS singles used a hypothetical fixed **−110** rather than a recorded spread price. 65/123 two-leg parlays and 45/57 three-leg parlays include at least one such assumed price. These cannot be treated as confirmed realized returns.
3. **Side match coverage is incomplete.** Of 324,920 odds rows meeting initial game filters, 149,227 rows could not be resolved to a home/away side and were dropped. Only 9,793 complete two-sided game/market/book pairs survived selection. Coverage varies dramatically by year, especially 2020.
4. **Retrospective input vintages.** Although per-game Elo is labeled pregame and training outcomes are walk-forward, no original contemporaneous Elo snapshots were independently archived. Future injury and player availability data were not used.
5. **Exposure/dependence.** Parlays assume cross-game independence; same-week game outcomes may not be independent. The price product is not a verified sportsbook parlay payout.
6. **Overfitting risk.** Ten seasons, and especially small parlay subsamples, have been inspected in model-building research. They are not untouched holdouts. Thresholds and stress factors are research assumptions, not calibrated statistical confidence bounds.
7. **No profitability claim.** The most relevant finding is that the straightforward singles engine lost roughly 6.32% in this historical replay. The positive parlay samples do not overturn that.

## Reproduce

```bash
pip install pandas numpy scikit-learn requests pytest
PYTHONPATH=. pytest -q tests/test_cfb_ten_year.py
PYTHONPATH=. python scripts/cfb_ten_year.py --output results/cfb_ten_year
```

GitHub Actions reruns on changes to the CFB script/tests/workflow, uploads the entire report and audit as an artifact, and force-adds selected results files to the repo despite the project's general `results/` gitignore.

Repository files: [report.json](../results/cfb_ten_year/report.json), [annual.csv](../results/cfb_ten_year/annual.csv), [singles.csv](../results/cfb_ten_year/singles.csv), [two-leg parlays](../results/cfb_ten_year/parlays_2.csv), [three-leg parlays](../results/cfb_ten_year/parlays_3.csv). The full two-sided market audit is in the workflow artifact.

## Next model improvements

For decision-grade evaluation, obtain pregame timestamped market snapshots, actual spread-side odds, native college advanced metrics (opponent-adjusted EPA/success rate/explosiveness), starter status and injuries captured before kickoff, and a fully out-of-time untouched evaluation season. Measure yearly market-only benchmarks and CLV. Only adjust features or calibration on training seasons; hold back recent data for final confirmation.
