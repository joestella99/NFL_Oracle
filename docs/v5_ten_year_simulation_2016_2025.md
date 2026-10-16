# NFL Oracle V5 — Ten-season ML / spread / mixed parlay simulation

**Completed:** October 16, 2026 | **Seasons:** 2016–2025 (10) | **Weeks:** 175 | **Games:** 2,628

**Verified passing GitHub Actions run:** https://github.com/joestella99/NFL_Oracle/actions/runs/38959831326

**Source code:** [scripts/parlay_v5_ten_year.py](../scripts/parlay_v5_ten_year.py), [scripts/parlay_mixed_market_v5.py](../scripts/parlay_mixed_market_v5.py), [config/parlay_mixed_market_policy.json](../config/parlay_mixed_market_policy.json).

**Artifact:** \`nfl-oracle-ten-year-replay\`, directory \`parlay_v5_ten_year/\`. Files: \`report.json\`, \`annual.csv\`, \`weekly.csv\`, \`by_parlay_length_and_market.csv\`, \`all_qualifying_tickets.csv\`, \`top20_each_week_including_rejected.csv\`, \`all_single_market_audits.csv\`.

**IMPORTANT INTERPRETATION:** "ROI" below is a *hypothetical equal-notional-unit return* on *every qualifying ticket*, at compounded historical NFLverse odds; it is **NOT** a real-money portfolio ROI or a forecast of gambling profit. The historical odds and spread prices are **not** verified as simultaneous, executable offers at one sportsbook, and cross-game legs/tickets have significant shared-team risk. This 2016–2025 period has been studied repeatedly when developing V1–V5, so it is **not** untouched holdout evidence.

## Exact approved V5 standards

- Favorable ML side >=60% predicted outright win probability.
- Best expected-value ATS side >=42% predicted cover probability.
- Minimum conservative probability all legs win (rather than push): 15%.
- Minimum conservative **ticket** EV +2%.
- Payout stress **1% per leg**.
- Conservative leg probability penalty: fixed 2 percentage points plus 20% of absolute model-vs-market probability difference.
- Two-, three- or four-leg **cross-game** parlays, featuring ML+ML, ML+ATS or ATS+ATS; never two legs of the same game.
- A spread push voids *only that leg* (it does not lose the entire ticket), including predicted push mass.
- No stake sizing, bankroll or dollar allocation. Every qualifying ticket is given exactly **one hypothetical unit** to calculate equal-unit return; additionally compute one highest conservative-EV eligible ticket per week as a more concentrated comparator.

## Aggregate results

| Probability source | Evaluated mathematical combinations | Qualified tickets | Wins | Ticket hit rate | Hypothetical equal-unit ROI | Weeks with a qualifying parlay | Return for best single eligible ticket/week |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Football without injury layer | 2,403,753 | 20,556 | 4,480 | 21.79% | **+13.30%** | 129/175 | **+18.26%** across 129 tickets |
| Football + time-filtered injury layer | 2,394,780 | 21,955 | 4,552 | 20.73% | **+7.40%** | 136/175 | +25.43% across 136 tickets |
| Market no-vig winner probability + ATS margin model | 2,444,485 | 8,988 | 1,816 | 20.20% | +4.73% | 118/175 | +12.39% across 118 tickets |

**Risk sensitivity (10-season block bootstrap of the football/no-injury equal-unit ROI, resampling NFL seasons): 5th–95th percentile approximately −5.32% to +44.85%.** Because negative returns are entirely plausible and the same years were used for research/model development, these are only descriptive stability checks, **not** legitimate forward ROI confidence bounds.

## Results by season — football without injury layer

| Season | Qualifying parlay tickets | Wins | Equal-unit ROI |
| --- | ---: | ---: | ---: |
| 2016 | 8,908 | 1,388 | **−13.01%** |
| 2017 | 2,673 | 763 | +54.28% |
| 2018 | 1,001 | 181 | **−25.34%** |
| 2019 | 1,727 | 624 | +87.48% |
| 2020 | 1,047 | 170 | **−22.18%** |
| 2021 | 3,331 | 856 | +29.58% |
| 2022 | 1,582 | 426 | +25.80% |
| 2023 | 157 | 34 | **−9.17%** |
| 2024 | 119 | 29 | +8.95% |
| 2025 | 11 | 9 | +260.08% |

**Critical:** 2025's huge +260% rate comes from **only 11 highly overlapping hypothetical tickets**. Qualifying volume collapsed dramatically after 2022. The aggregate football model's apparently positive ROI is fragile and **does not prove present-day profitability**.

### Type breakdown — football without injury layer

| Parlay market type | Qualifying tickets | Equal-unit ROI |
| --- | ---: | ---: |
| Moneyline only (2–4 legs) | 1,420 | **+51.07%** |
| ML + ATS mix (2–4 legs) | 16,706 | +12.16% |
| ATS only (2–4 legs) | 2,430 | **−0.88%** |

| Leg count | Qualifying tickets | Equal-unit ROI |
| --- | ---: | ---: |
| 2 legs | 3,335 | **+0.59%** |
| 3 legs | 11,087 | +12.05% |
| 4 legs | 6,134 | +22.49% |

These are **retrospective subgroups**, not independent or causally validated strategies. Do not optimize the next model to the best-looking subgroup without prospective validation.

## Recent-year warnings

- Football model during 2023–2025: **287** qualifying tickets across three seasons, hypothetical **+8.66%** equal-unit ROI, mostly flattered by 2025's 11-ticket sample. Restricting to 2023–2024 yields **276** tickets with **−1.36%** ROI.
- Injury-enhanced model in 2024: **1,519** qualifying tickets, only **102** wins, **−72.09%** equal-unit ROI. The injury layer produced far more supposedly qualified bets while harming forecast quality. **Do not promote the injury-enhanced variant** without resolving its selection/calibration failure.
- 2025 football model: only 11 qualified bets, giving insufficient evidence even for a stable annual ROI estimate.
- Market no-vig model: zero qualifying **ML-only** parlays over the entire ten years; its 8,988 qualifications are mixed/ATS and originate in the empirical margin model. A no-vig probability combined with vigged moneyline quotes **cannot** create genuine ML positive EV by merely multiplying favorites.

## Methodology

- 2014–2015 **strictly held-out score-margin forecasts** were generated to establish a residual distribution available *before* 2016.
- Historical margin projections for 2019–2025 reused the frozen walk-forward margin model; 2014–2018 warmup projections refit the same method.
- Every year's ATS probability distribution was calculated strictly from **previous seasons'** score-margin errors, never the current evaluated season.
- Each year's football/injury winner probability model refitted from previous seasons' game-by-game out-of-sample forecasts. No future outcome used in selection.
- A week includes all qualifying parlay combinations; these are heavily overlapping and not unique independent bets. The "one best eligible ticket per week" comparator is supplied separately.
- Historical pricing and correlation assumptions are the largest caveats; even an otherwise leak-free retrospective scoring script is not a real-world execution backtest.
- The official V5 research selection criteria are unchanged by this simulation.

## Conclusions

1. V5's broader ML+ATS combinatorics *increase research coverage*, but positive historical ROI does **not** necessarily establish edge.
2. ATS-only fares approximately flat/negative historically; many profitable-looking ticket subgroups are driven by ML legs and long-shot payouts.
3. The injury model's **2024 failure (−72%)** and the selection collapse in **2023–2025** demand calibration/outcome attribution, not another arbitrary loosening of thresholds.
4. Validate on **future 2026–2027 frozen, time-stamped same-book prices** (including actual quoted parlays) and a finite set of prespecified risk/selection variants. Only prospective evaluation can inform whether this represents a repeatable advantage.
