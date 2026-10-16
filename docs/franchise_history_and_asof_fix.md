# Franchise history and point-in-time corrections

**Status: proposed research fix, not a profitable strategy.**

The 2016-2025 evaluation includes Rams (STL → LA), Chargers (SD → LAC),
and Raiders (OAK → LV) relocations. The older code keyed rolling scores,
EPA, scheme matchups, and offensive consistency by raw team abbreviations.
This discarded relevant earlier franchise history. The new
`franchise_key` applies **only to rolling-team history and PBP joins**.
Original IDs, sportsbook names, odds quotes, and schedule abbreviations
are not rewritten. LA and LAC are never conflated.

The QB/EPA builder also skipped the explicit completed-score check
already used by other feature families. When schedule scores are
available, in-progress PBP is now excluded from history updates.
Feature construction remains same-day isolated.

## Earlier local diagnostics — not reproduced by this PR

The earlier uncommitted local correction (October 16) reportedly changed
V5 all-value hypothetical ROI from **+13.30% to -1.97%**. V6 all-confidence
ROI improved from **-8.32% to -2.36%**, still negative. The 464 priority
tickets / 145 weeks experiment was +3.17% hypothetical ROI. These are
development-sample diagnostics, **not** prospective evidence and **not**
confirmed by a new CI replay. Do not represent the old +13.30% as valid
after this correction; regenerate comparisons before considering policies.

## Next gate before selecting a forecasting model

1. Run the complete, corrected 2016–2025 walk-forward replay and report
   model and market Brier, log loss, calibration by probability bucket,
   margin error and annual parlay hit/ROI for 2023–2025 separately.
2. Compare a football-based pregame margin and market-residual forecast
   against an actual market-only baseline using earlier-season
   calibration; do not promote the previously unsuccessful prototype
   on development ROI.
3. Independently capture pre-kickoff QB, injury and same-book prices for
   prospective 2026 forecasts, with missing inputs explicitly marked.
4. Preserve both V5 value and V6 confidence/preferred views. More
   ticket volume cannot replace probability calibration or edge.

**No bankroll/dollar allocation is introduced.** Historical multiplied
leg prices are not verified accepted same-book parlay offers, and
all research seasons have been reused for model development.
