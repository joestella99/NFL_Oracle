"""Canonical NFL Oracle parlay selection policy (v3), experimental backtest.

Read the official, versioned selection configuration. Legacy v2 and frozen
straight/parlay checkpoint remain available for independent comparisons.

Run:
PYTHONPATH=src:. python scripts/parlay_portfolio_v3.py \
  --predictions results/ten_year_comparison/predictions.csv \
  --warmup results/ten_year_comparison/warmup_predictions_2014_2015.csv \
  --straight-bets results/ten_year_ml_spread_parlays/straight_bets.csv \
  --old-parlays results/ten_year_ml_spread_parlays/calculated_cross_game_parlays.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from scripts.parlay_portfolio_v2 import RiskThreshold, evaluate

DEFAULT_POLICY_FILE = Path(__file__).resolve().parent.parent / "config" / "parlay_policy.json"
DEFAULT_OUTPUT = "results/parlay_portfolio_v3"


def load_policy(path: Path = DEFAULT_POLICY_FILE) -> tuple[dict, RiskThreshold]:
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get("status") != "official_default_parlay_selection_policy":
        raise ValueError("Refusing nonofficial parlay configuration")
    risk = RiskThreshold(**descriptor["risk"])
    risk.validate()
    if risk.require_positive_leg_ev:
        raise ValueError("V3 selection must NOT impose a hard individual-leg EV veto")
    if risk.maximum_total_game_risk_fraction is None:
        raise ValueError("V3 selection must use separate parlay and total-game caps")
    if risk.per_leg_price_haircut != .01 or risk.min_conservative_ev != .02:
        raise ValueError("V3 official minimum EV + price haircut must be 2% / 1%")
    return descriptor, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--warmup", required=True)
    ap.add_argument("--straight-bets", required=True)
    ap.add_argument("--old-parlays", required=True)
    ap.add_argument("--policy-config", default=str(DEFAULT_POLICY_FILE))
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    descriptor, risk = load_policy(Path(args.policy_config))
    forecasts = pd.concat(
        [pd.read_csv(args.warmup), pd.read_csv(args.predictions)],
        ignore_index=True,
    )
    report, weekly, annual, candidates = evaluate(
        forecasts, pd.read_csv(args.straight_bets),
        pd.read_csv(args.old_parlays), risk=risk,
    )
    report["policy_id"] = descriptor["policy_id"]
    report["policy_status"] = descriptor["status"]
    report["selection_rule"] = descriptor["description"]
    report["comparison_uses_same_2016_2025_seasons"] = True
    report["unverified_bookmaker_prices"] = True
    report["not_auto_betting"] = True
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    weekly.to_csv(root / "weekly.csv", index=False)
    annual.to_csv(root / "annual.csv", index=False)
    candidates.to_csv(root / "all_combinations_and_exposure.csv", index=False)
    print("OFFICIAL PARLAY SELECTION POLICY", descriptor["policy_id"], flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(annual.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
