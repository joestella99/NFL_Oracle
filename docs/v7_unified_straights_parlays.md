# V7 — unified straight picks and cross-game parlays

**Status:** research branch awaiting CI and corrected ten-season replay, not a proven betting strategy.

## What changed

V7 uses one probability pipeline to examine all four game markets: both
moneylines and both ATS sides. It does not discard a winner or underdog
because an older shortlist chose the wrong side. Straight picks are ranked
by probability, raw expected value and a fixed stress scenario. High
confidence but badly priced picks remain visible and are not described as
profitable.

For season Y the engine calibrates a constrained blend of no-vig market and
football win estimates using strictly earlier seasons of out-of-sample
forecasts, selecting the blend on Brier score with a small zero-weight
penalty. It similarly learns how much of the football expected margin to
use over the sportsbook point spread, comparing prior-year margin MSE.
Prior-year calibrated residuals yield ATS probabilities and explicit push
rates. The adjusted moneyline and ATS inputs feed the *same* straight and
parlay generation pipeline.

For parlay exploration keep best-value ML, best-value ATS, and strongest
high-confidence alternative per game (at most three total). Enumerate
cross-game combinations of two or three legs by default, within a single
sportsbook and a one-hour maximum quote timestamp skew. All combinations
remain auditable; a small edge alone is not a veto. Negative-edge legs
are exposed as experimental and never mislabeled profitable. Multi-ticket
feature rankings penalize game overlap to avoid showing five copies of
one opinion. No stakes, bankroll restrictions or wager execution.

Likes only affect the preferred display board and the explicit liked-pick
audit, never probabilities. V5 and V6 remain available for comparison.

## Settlement, notional returns and caveats

Straight expected net return = p_win * decimal_odds + p_push - 1.
Parlay expected net return = product(each p_win * decimal_odds + p_push) - 1.
ATS pushes void only their leg. The stress score subtracts probability
points and discounts price; it is NOT a statistical confidence interval.
Prior seasons were used for development, and historical compounded odds
have NOT been verified as simultaneously executable same-book tickets.

## Historical replay

After producing the existing out-of-sample winner and margin artifacts:

    PYTHONPATH=src:. python scripts/decision_v7_replay.py --predictions results/ten_year_comparison/predictions.csv --warmup results/ten_year_comparison/warmup_predictions_2014_2015.csv --recent-margins results/margin_model_comparison/margin_predictions.csv

Annual output under results/decision_v7 includes calibration Brier/log loss,
past-only learned model weights, all straight sides, parlay audit, one-per-week
historical ticket comparisons, and a calibration input CSV. Compare later
2023–2025 outcomes to the market baseline; do not optimize thresholds after
seeing each tested season.

## Live usage

    PYTHONPATH=src:. python scripts/parlay_live_mixed.py --strategy v7 --as-of 2026-10-16T16:00:00-04:00 --quotes-csv same_book_quotes.csv --prior-margins results/margin_model_comparison/margin_predictions.csv --calibration-csv results/decision_v7/oos_calibration_inputs.csv --liked-picks liked_picks.csv

The nfl-oracle shortlist command uses the same V7 engine. If no historical
OOS calibration CSV is supplied, live moneyline forecasts intentionally
fall back to market-only probabilities, clearly flagged as uncalibrated.
Observed quote timestamps do not prove the sportsbook would accept the
calculated parlay. No positive prospective ROI is asserted.
