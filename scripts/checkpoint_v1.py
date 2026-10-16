"""NFL Oracle v1 research checkpoint: frozen historical winner.

Do not silently choose a new variant based on the same evaluation years.
Retains the risk_tolerant $100/week uncapped straight allocator and separate
old one-ticket calculated cross-game parlay sleeve. This is a reproducible
2016-2025 *historical* checkpoint, NOT a validated live betting system.

Run after scripts/simulate_ten_year_portfolio.py produces its outputs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

CHECKPOINT_ID="nfl-oracle-research-v1-2026-10-16"
POLICY="risk_tolerant"
WINDOW=list(range(2016,2026))
FILES=("straight_bets.csv","calculated_cross_game_parlays.csv",
       "annual_comparison.csv","weekly_comparison.csv","ten_year_totals.csv")


def freeze(root:Path,destination:Path):
    missing=[name for name in FILES if not (root/name).exists()]
    if missing:raise FileNotFoundError(f"Missing historical outputs: {missing}")
    straight=pd.read_csv(root/"straight_bets.csv")
    parlay=pd.read_csv(root/"calculated_cross_game_parlays.csv")
    annual=pd.read_csv(root/"annual_comparison.csv")
    weekly=pd.read_csv(root/"weekly_comparison.csv")
    total=pd.read_csv(root/"ten_year_totals.csv")
    if not set(WINDOW).issubset(set(straight.season)):
        raise AssertionError("Missing full 2016-2025 history")
    if "risk_tolerant" not in set(total.policy):raise AssertionError("Missing canonical strategy")
    if straight.duplicated(["policy","game_id"]).any():
        raise AssertionError("More than one straight bet on a game")
    risk=straight.loc[straight.policy.eq(POLICY)].copy()
    p=parlay.loc[parlay.policy.eq(POLICY)].copy()
    check=risk.groupby(["season","week"]).stake.sum()
    if len(check)!=175 or not ((check-100.).abs()<.005).all():
        raise AssertionError("Every one of 175 weeks must spend exactly $100 on straights")
    if p.duplicated(["season","week"]).any():
        raise AssertionError("Canonical parlay strategy is at most one ticket weekly")
    week=weekly.loc[weekly.policy.eq(POLICY)].copy()
    if len(week)!=175:raise AssertionError("Missing weekly comparison rows")
    stake=float(risk.stake.sum()+p.stake.sum())
    profit=float(risk.profit.sum()+p.profit.sum())
    summary=total.loc[total.policy.eq(POLICY)].iloc[0]
    if abs(stake-float(summary.combined_staked))>.02:
        raise AssertionError("Total bet size disagrees with checkpoint")
    if abs(profit-float(summary.combined_net))>.02:
        raise AssertionError("Realized historical P/L disagrees with checkpoint")
    destination.mkdir(parents=True,exist_ok=True)
    risk.to_csv(destination/"canonical_straights.csv",index=False)
    p.to_csv(destination/"canonical_parlays.csv",index=False)
    annual.loc[annual.policy.eq(POLICY)].to_csv(destination/"year_by_year.csv",index=False)
    week.to_csv(destination/"week_by_week.csv",index=False)
    manifest={
        "checkpoint_id":CHECKPOINT_ID,
        "policy":POLICY,"status":"frozen historical research baseline",
        "evaluated_years":[WINDOW[0],WINDOW[-1]],
        "week_count":int(len(check)),
        "weekly_straight_budget":100,
        "bets_per_week_cap":None,
        "straight_market_selection":"highest qualifying adjusted EV per distinct game; forced highest-probability ML when none",
        "parlay_selection":"one optional 2-leg ML cross-game parlay from rejected opportunities",
        "parlay_stake":"extra $2-$15, subject to historical experimental selector",
        "parlay_ev_warning":"experimental selector previously allowed slightly negative estimated EV, not a proven value screen",
        "total_staked":round(stake,2),
        "net_profit":round(profit,2),
        "roi_pct":round(100*profit/stake,4),
        "single_bets":int(len(risk)),"parlay_tickets":int(len(p)),
        "research_alternatives":"Retained separately in scripts/parlay_round_robin.py, scripts/selective_parlays.py and scripts/opportunity_breadth.py",
        "not_live_promoted":True,
        "limitations":[
            "Historical moneyline quotes are not verified as simultaneous same-sportsbook executable prices.",
            "Backtest years were reused repeatedly for model/selection design; no untouched holdout.",
            "Cross-game parlay probability multiplication assumes independence.",
            "Parlay staking can be negative estimated value; profits were observed retrospectively.",
            "Historical positive ROI does not guarantee future profitability.",
            "The live shortlist model differs from the full historical model.",
        ]
    }
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (destination/"README.md").write_text(
        "# NFL Oracle research checkpoint v1\\n\\n"
        "This checkpoint freezes the historically best-tested risk-tolerant "
        "moneyline/ATS plus separate two-leg parlay configuration. "
        "It is NOT a production betting strategy or guaranteed to profit.\\n\\n"
        "Run the ten-year backtest workflow and compare subsequent strategies "
        "against `manifest.json` and these exact selection files. "
        "Do not retroactively replace this baseline when a new experiment "
        "does worse.\\n"
    )
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="results/ten_year_ml_spread_parlays")
    parser.add_argument("--output",default="results/checkpoint_v1")
    args=parser.parse_args()
    print(json.dumps(freeze(Path(args.source),Path(args.output)),indent=2))


if __name__=="__main__":main()
