# NFL Oracle V7: guarded decisions (October 23, 2026)

## Scope and status

V7 is **a new research-only decision layer**, not a replacement for the V5/V6
historical forecasting model. It does not claim a better NFL winner predictor
or verified profitability. V5/V6 checkpoints are unchanged.

The October 18 Colts/Patriots/Ravens recommendation was a market-based manual
shortlist, not a V6 model run. Do not treat those outcomes as measured V6 errors.

## What is implemented

- `src/nfl_oracle/v7.py`: chronological historical win-probability bucket
  reliability adjustment with mandatory sample count and a conservative
  probability stress. If insufficient samples: **ABSTAIN**, not a guess.
- Both sides of both markets can be evaluated (no favorite-only exclusion).
- Every promoted leg must clear a *conservative single-leg EV* threshold.
- Cross-game two/three-leg combination auditing groups by bookmaker, season,
  week, and model; never puts both sides of a game on the same ticket.
- Probability and price stresses remain declared **heuristics**, not claimed
  confidence intervals. Different games are treated as independent only for
  research expected-value calculations.
- All rejected and qualified combinations remain visible.

## Running (after installing repo dependencies)

First generate a pregame audit with `scripts/parlay_live_mixed.py`.
That script writes `market_side_audit.csv`, including both home and away sides
with `won` unknown for upcoming matches. Append an explicit `bookmaker`
column to both live and historical audits if not already present.

The `--history` file must contain genuine **earlier** as-of observations
with `season,week,model,kind,win_probability,won`. An observed `won` must
refer to the selection's final settlement; do not fill unplayed games. Season
and week cutoffs are enforced again inside V7.

```bash
PYTHONPATH=src:. python scripts/v7_decisions.py \
  --legs results/parlay_pregame_mixed/market_side_audit.csv \
  --history results/calibration_history.csv \
  --season 2026 --week 7 \
  --output results/v7_week7
```

Outputs: `legs_audit.csv`, `tickets_audit.csv`, `report.json`.

## Known blocking gaps before any actual wager recommendations

1. The current live forecast is still a simplified team-form model, not the
   full historical EPA/QB/matchup model. This must be unified before relying on
   historical claims.
2. NFLverse historical prices aren't verified synchronized sportsbook quotes.
   Accepted parlay payouts may differ from multiplied leg prices.
3. Time-stamped injury/starter coverage is inadequate for reliable model claims.
4. The probability adjustment is a coarse historical bucket diagnostic.
   It should be replaced by walk-forward fitted calibration with documented
   uncertainty coverage, and checked against the market-only baseline.
5. Historic stratified calibration requires settled leg-side observations
   recorded at the **original pregame forecast snapshot**, not reconstructed
   from postgame market information.
6. Prospective backtests with model-vs-market comparisons and drawdown analyses
   are needed; do not treat 2016–2025 as untouched test data.

## Critical execution policy

No model run, valid prices, or adequate historical calibration means **NO
RECOMMENDATION**. V7's `RESEARCH_QUALIFIED` flag is intentionally not a
betting recommendation. The policy is percentage-based and does not allocate
stakes or place wagers.

## Integrated live invocation

The existing live-preview entrypoint now supports an optional `--v7-history`
argument. It keeps the established V6 outputs, additionally exporting
`v7_guarded_legs.csv` and `v7_guarded_tickets.csv` for each season/week,
plus V7 research funnel metrics in the report.

```bash
PYTHONPATH=src:. python scripts/parlay_live_mixed.py \
  --as-of 2026-10-23T13:00:00-04:00 \
  --quotes-csv verified_same_book_quotes.csv \
  --prior-margins earlier_margin_predictions.csv \
  --v7-history saved_prior_week_market_side_audits.csv
```

Inputs are still required; this does not fetch missing sportsbook data or
make retroactively known results available before they happened. Historical
calibration is computed from earlier weeks/seasons only. The live predictor
remains preliminary rather than the full historical EPA/QB ensemble.

