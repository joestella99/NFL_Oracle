> **PROMOTED POLICY — October 16, 2026:** The user selected the sensitivity variant with conservative **combined ticket EV >= +2%**, **1% gross-price haircut per leg**, no mandatory individual-leg EV veto, a separate parlay-game cap ($20) and combined straight+parlay game cap ($120). The official default is now `scripts/parlay_portfolio_v3.py` using `config/parlay_policy.json`. The older variant remains available for historical comparison, and these retrospective ROI figures remain exploratory.

# Parlay V2 selection standards audit — October 16, 2026

**Research only.** Source: successful run [38948326895](https://github.com/joestella99/NFL_Oracle/actions/runs/38948326895), `nfl-oracle-ten-year-replay` artifact, `parlay_portfolio_v2/all_combinations_and_exposure.csv`, `ten_year_ml_spread_parlays/straight_bets.csv`. All 2016–2025 years were previously inspected in development. The odds are historically compounded, **not** verified as executable at one bookmaker and instant.

## The funnel (175 NFL weeks)

- Enumerated **526,308** candidate tickets, comprising 8,465 two-leg, 25,311 three-leg, 53,166 four-leg, and 439,366 five-to-sixteen-leg combinations. These reuse games extensively: 526,308 is not the number of independent betting opportunities.
- First-decision rejection: **526,214 LEG_NOT_VALUE** (at least one leg fails conservative individual EV; many of these also fail later gates), **5 low conservative joint probability**, **2 below +2% ticket EV**; **87 qualify** and **46 funded**.
- **41 of 87** eligible tickets receive no money because the $20 *combined* game cap is already exhausted by the separate existing straight portfolio, not because the ticket itself fails the parlay value screen.
- **43 of 46 funded** are two-leg; **3** three-leg, zero four-leg-plus.
- Under the strict original rule only **23 of 175 weeks** even contain a qualifying ticket. **2024 and 2025: zero funded**, and indeed **zero tickets passing both +2% conservative ticket EV and 15% joint probability even after eliminating the individual-leg EV rule**.

## Sensitivity study (fixed 2016–2025 predictions and outcomes)

All candidate selections use only forecast probabilities and odds before looking at outcomes. A fixed $1,000 simulated bankroll, 5% weekly parlay risk, $5 maximum per ticket and $20 maximum parlay exposure to each game. "Separate parlay allowance" adds a $120 ceiling for combined existing-straight-plus-parlay *game* risk, instead of applying the $20 game cap directly to existing straight positions. This is a **research accounting hypothesis**, not a policy endorsement. Gross payouts are from the same unverified historical leg quotes; quoted price stress is applied at selection, **not** to realized payoff.

| Historical screen / allocation | Candidate tickets | Funded | Total stake | Profit | ROI on stake | 2023–2025 profit | 2024–25 funded |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 strict and $20 shared game cap | 87 | 46 | $186.97 | +$66.39 | +35.51% | -$5.00 | 0 |
| Same strict value test; separate $20 parlay-game cap and $120 combined cap | 87 | 62 | $298.35 | +$109.34 | +36.65% | -$5.00 | 0 |
| Combined-ticket conservative EV >= 2%, joint >= 15%, no individual-leg gate; separate limits | 848 | 132 | $600.38 | +$207.91 | +34.63% | -$13.73 | 0 |
| Combined-ticket conservative EV >= 0%, joint >= 15%, no individual-leg gate; separate limits | 1,075 | 195 | $705.22 | +$213.33 | +30.25% | -$14.35 | 0 |
| 1%/leg instead of 2% price stress; ticket EV >= 2%, joint >= 15%, separate limits | 1,259 | 199 | $921.44 | +$238.03 | +25.83% | -$32.41 | 0 |
| 1%/leg stress, ticket EV >= 0%, joint >= 15%, separate limits | 1,634 | 256 | $1,002.56 | +$246.70 | +24.61% | -$35.27 | 3 |

**Do not optimize to whichever line wins this old sample.** All alternatives were tested on the same previously-inspected seasons. Profit is concentrated in older years; opening the gates increases total stake and retrospective profit but the 2023–2025 results get worse. No positive ROI or high wager count is demonstrated for fresh seasons.

The football-specific predictor's reported historical Brier score is **0.211355**, worse than the market's **0.210522**. Therefore apparent ticket EV from that model is not yet a validated edge.

### Screening accounting and double-conservatism

1. `base_probability_haircut=.02` removes 2 points from every leg's modeled win probability, then `disagreement_haircut_weight=.20` removes still more for disagreement and model-vs-market differences.
2. `per_leg_price_haircut=.02` multiplies *gross* ticket payout by 0.98 on **every leg**, so a four-leg payout receives approximately a 7.76% additional reduction at the ticket-value test.
3. Every leg must *also* have positive conservative EV individually before considering the ticket. This "weakest leg veto" drives almost the entire rejection funnel; it should be an explicit configurable policy and comparison, not an unexplained hard-coded prerequisite. A marginally negative-value leg ordinarily makes a parlay less efficient than omitting it, however, so removing the veto is **not automatically superior**.
4. The 2%-per-leg quote penalty was not empirically estimated from verified simultaneous bookmaker quotes. It is a stress-test assumption. If actual payout is 2% per leg worse, the original 46 funded tickets' hypothetical profit decreases from **+$66.39 to +$56.11** (approximately 30.01% ROI); it is inappropriate to present the optimistic unstressed return as executable.
5. Existing straight stakes consume the **entire** $20 game cap, inadvertently nullifying the stated feature "parlays may overlap straight picks" in many cases. Proper accounting should independently cap the parlay sleeve *and* total game risk across all ticket types, rather than ignoring or incorrectly conflating either.

## Proposed standards for NEXT RESEARCH iteration

- **Do not set a target acceptance rate** or force a parlay every week. Count source games, tickets passing *each gate*, eligible distinct games, funded tickets and risk-block reasons separately.
- Keep one selection per game and prohibit conflicting sides of a game; allow comparison of underdog sides when odds offer positive modeled value, not only favorites with >60% hit rates. Require valid contemporaneous sportsbook quotes before any live recommendation.
- Calculate both base and conservative calibrated joint win probability. Keep the ≥15% conservative joint-probability research floor for now; sensitivity-test it, rather than loosening to hit volume.
- Run parallel `strict per-leg positive EV` and `ticket-only positive conservative EV` watchlists with the same ≥2% ticket-EV hurdle. **Do not silently promote the liberal variant to funded/live** while the underlying probabilities fail to outperform market calibration. Record minimum per-leg EV so weaknesses remain visible.
- Compare 1% versus 2% quote deterioration and base probability haircuts in sensitivity reports; do not assume the haircuts are real confidence intervals.
- Fix **exposure accounting**: budget caps for all parlays together, per parlay ticket, per *parlay-game* shared across tickets, and per game across straight+parlay. A candidate can be funded on several tickets only when each cap has remaining room.
- Add outcomes by season, forecast-probability bucket, odds bucket, leg count, game overlap, market-vs-model divergence, candidate close-line value, drawdowns and risk-adjusted returns. **Prioritize 2026 forward frozen out-of-sample, timestamped lines**, not picking thresholds off the repeatedly-tested 2016–2025 sample.
- The production/legacy parlay engine remains untouched. These are sensitivity findings, not any placed wagers, guaranteed profitability, or bankroll allocation instructions.

## Reproduction

Download the `nfl-oracle-ten-year-replay` artifact for the linked GitHub Actions run. `parlay_portfolio_v2/all_combinations_and_exposure.csv` provides every ticket's `season`, `week`, `games`, `legs`, `conservative_probability`, `calculated_decimal_odds`, `conservative_ev`, `status`, `stake`, and `won`; `ten_year_ml_spread_parlays/straight_bets.csv` provides the original straight allocations. For an alternative price haircut h, recalculate ticket conservative EV as `conservative_probability * calculated_decimal_odds * (1-h)**legs - 1`. Sort eligible tickets by `conservative_ev/(stressed_decimal_odds-1) * sqrt(conservative_probability)`, allocate greedily in whole cents subject to the same weekly/ticket/parlay-game/combined-game/Kelly fraction caps and a $0.25 minimum. Evaluate outcomes *after* selection. Stake and ROI results may differ slightly with alternative tie-breaking/rounding; every reported historical figure is exploratory.
