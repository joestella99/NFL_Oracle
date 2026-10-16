# V6: high-confidence, small-edge parlays

V5 already admits small-edge legs to combination search. Its final +2% conservative ticket EV filter and EV-first leaderboard can hide modest-return, higher-hit-rate tickets. V6 adds a confidence view and liked-pick audit alongside the existing value view. This is a research rebuild, not a proven profitable strategy.

## Rules

| View | Eligibility | Ordering |
| --- | --- | --- |
| Confidence | Each leg >=65% outright/cover probability and >=0% raw model EV; 2–3 distinct games; ticket >=40% full-payout win probability and >=0.5% raw model EV | Highest full-payout probability, then EV, then shorter length |
| Value | Original V5 candidate pool, probability stress, +2% stressed ticket EV, 2–4 legs | Highest stressed EV |
| Preferred | Confidence-qualified tickets containing explicitly liked picks | Number of liked selections, then confidence and EV |

Defaults were chosen before the V6 replay, without a threshold sweep. They are experimental, not calibrated confidence bounds. No payout target, compulsory ticket, dollar allocation, or forced inclusion. Liking a pick never changes its probability or eligibility. An excluded liked pick remains in `liked_pick_audit.csv`, with a reason: missing quotes, failed leg standard, or no qualifying partner.

ML and spread picks remain eligible. All shorter subsets are evaluated; no leg is appended just to increase payout. The confidence view marks the probability/EV Pareto frontier. Rankings do not allocate money or imply every combination should be played. The one-ticket-per-week leader is a diagnostic, not a compulsory wager or portfolio optimizer.

## Why small edges can work

For independent legs without pushes, ticket EV is `product(p_win * decimal_odds) - 1`. Two +1% EV legs produce +2.01% ticket EV. Modest individual edges do not need to clear a standalone-bet threshold.

A zero-edge leg leaves expected gross return unchanged while reducing full-payout probability. A negative-edge leg lowers expected gross return. A parlay does not manufacture value from those legs. The confidence view rejects negative-edge legs; the separate frozen V5 value view retains its original rules.

Each ticket exports per-leg marginal diagnostics: EV without each leg, EV change from adding it, full-payout probability change, the leg probability required for the entire ticket to break even, and the remaining probability cushion. ATS pushes preserve `product(p_win * decimal + p_push) - 1` expected-return math and void only their own leg on settlement.

Raw and stressed EV are both visible. Confidence tickets failing V5 stress are labelled `POSITIVE_MODEL_EV_STRESS_FAILS`. Tickets passing both views receive `ROBUST_UNDER_V5_STRESS`, describing only that particular stress test.

## Ten-season diagnostic: 2016–2025

Inputs: frozen `parlay_v5_ten_year/all_single_market_audits.csv` and `all_qualifying_tickets.csv` from [Actions run 38959831326](https://github.com/joestella99/NFL_Oracle/actions/runs/38959831326), artifact 12072163845. V6 reuses prior-year winner forecasts and prior-year ATS residual probabilities without refitting on test outcomes. Outcomes do not determine ranking or inclusion.

Football model without injury layer:

| Selection diagnostic | Tickets | Hit rate | Hypothetical equal-unit ROI |
| --- | ---: | ---: | ---: |
| V5 all value-qualified tickets | 20,556 | 21.79% | +13.30% |
| V5 highest-value ticket per qualifying week | 129 | 20.16% | +18.26% |
| V6 all confidence-qualified tickets | 793 | 44.39% | -8.32% |
| V6 highest-confidence ticket per qualifying week | 126 | 56.35% | +7.80% |
| V6 confidence tickets failing V5 stress | 664 | 45.48% | -9.53% |
| V6 tickets with a 0% to <2% edge leg | 535 | 46.54% | -7.18% |

The confidence pool contains 543 two-leg and 250 three-leg tickets. All 126 confidence leaders were two-leg tickets, emerging from probability-first ordering rather than a two-leg-only rule. Of those leaders, 106 failed V5 stress.

Leaders averaged 58.99% predicted full-payout probability versus a 56.35% observed positive-return rate. The complete confidence pool averaged 53.20% versus 44.39%, exposing substantial overconfidence/selection error. With spread pushes, positive-return and full-payout events can differ.

Recent results limit the positive headline: confidence leaders returned **+0.28% over 36 tickets in 2023–2025**, and **-18.56% over nine tickets in 2025**. Taking every confidence candidate has not demonstrated an advantage. The no-vig market model produces zero confidence-qualified tickets; compounding sportsbook favorites does not manufacture edge. The injury variant remains a separate comparator.

All results compound historical individual-leg prices, not accepted same-book parlay quotes. Combinations overlap, and these seasons were already inspected for development. This is retrospective research, not untouched holdout evidence or a live-profit claim. The full tables, including market and injury comparators, are [comparison](v6_conviction_comparison.csv) and [annual results](v6_conviction_annual.csv).

## Usage

The live-preview CLI defaults to V6; same-book, timestamp-checked individual ML and ATS prices remain mandatory. Optional liked-picks CSV:

```csv
game_id,side,kind
2026_06_ARI_LA,home,ML
2026_06_LAC_KC,home,ML
```

```sh
PYTHONPATH=src:. python scripts/parlay_live_mixed.py \
  --as-of 2026-10-16T16:00:00-04:00 \
  --quotes-csv current_same_book_quotes.csv \
  --prior-margins prior_season_margin_predictions.csv \
  --liked-picks liked_picks.csv
```

Outputs: `confidence_view.csv`, `value_view.csv`, `preferred_view.csv`, `liked_pick_audit.csv`, complete ticket/leg audits and weekly funnel. Use `--strategy v5` for the value-only view. The live football alternative remains the existing pregame-form model, not the historical full football/injury ensemble. These historical returns are not the live model's track record.

Replay existing artifacts:

```sh
PYTHONPATH=src:. python scripts/parlay_conviction_v6.py \
  --leg-audit results/parlay_v5_ten_year/all_single_market_audits.csv \
  --v5-tickets results/parlay_v5_ten_year/all_qualifying_tickets.csv
```

The Actions workflow runs the comparison after V5 and retains its outputs. Local validation: **120 tests passed**, covering small-edge admission, bad-price rejection, marginal EV, pushes, preference integrity, V5 pool preservation, outcome-independent ranking, and the complete model → quotes → upcoming ticket pipeline across two books. That full flow also exposed and fixed existing missing-moneyline and hard-coded model-variant preview failures.
