"""True 2016-2025 NFL Oracle portfolio replay: straights vs parlays vs total.

2014-2015 are walk-forward out-of-sample calibration-only warmup years;
2016-2025 are the 10 evaluated seasons. Training never observes a
game outcome from the year under prediction. Historical betting quotes
are not time-stamped, so returns describe a mathematical replay only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from nfl_oracle.cli import get_data
from scripts.margin_model_comparison import margin_replay
from scripts.roi_decision_v2 import candidates as build_candidates
from scripts.roi_full_budget_parlays import (
    make_singles, parlay_from_rejected, STRAIGHT_BUDGET, PARLAY_MAX_BUDGET,
)

EVALUATED_SEASONS = tuple(range(2016, 2026))


def _summary(straights: pd.DataFrame, parlays: pd.DataFrame, season: int | None = None):
    s = straights if season is None else straights.loc[straights.season.eq(season)]
    p = parlays if season is None or parlays.empty else parlays.loc[parlays.season.eq(season)]
    if parlays.empty:
        p = parlays
    sw = float(s.stake.sum())
    sp = float(s.profit.sum())
    pw = float(p.stake.sum()) if len(p) else 0.
    pp = float(p.profit.sum()) if len(p) else 0.
    weeks = s[["season","week"]].drop_duplicates().shape[0]
    # Show both actual pooled-stake ROI and profit per available $100 weekly.
    return {
        "season": "2016-2025" if season is None else int(season),
        "weeks": int(weeks),
        "straight_bets": int(len(s)),
        "straight_ml": int(s.kind.eq("ML").sum()),
        "straight_spread": int(s.kind.eq("ATS").sum()),
        "straight_staked": round(sw, 2),
        "straight_net": round(sp, 2),
        "straight_roi_pct": round(100*sp/sw, 3) if sw else None,
        "straight_positive_weeks": int((s.groupby(["season","week"]).profit.sum()>0).sum()),
        "fallback_week_count": int(s.loc[s.forced_fallback,["season","week"]].drop_duplicates().shape[0]),
        "parlay_tickets": int(len(p)),
        "parlay_staked_extra": round(pw, 2),
        "parlay_net": round(pp, 2),
        "parlay_roi_pct": round(100*pp/pw, 3) if pw else None,
        "parlay_wins": int(p.both_won.sum()) if len(p) else 0,
        "combined_staked": round(sw+pw, 2),
        "combined_net": round(sp+pp, 2),
        "combined_roi_pct": round(100*(sp+pp)/(sw+pw), 3) if sw+pw else None,
        "profit_per_100_weekly_budget_pct": round(100*(sp+pp)/(STRAIGHT_BUDGET*weeks), 3) if weeks else None,
    }


def weekly_comparison(singles, parlays):
    """One row per policy/week: straight, parlay, and combined P/L and ROI."""
    keys=["policy","season","week"]
    base=singles.groupby(keys).agg(
        straight_bets=("game_id","size"),
        straight_ml=("kind",lambda x:int((x=="ML").sum())),
        straight_spread=("kind",lambda x:int((x=="ATS").sum())),
        straight_staked=("stake","sum"),straight_net=("profit","sum"),
        forced_fallback=("forced_fallback","max"),
    ).reset_index()
    if len(parlays):
        add=parlays.groupby(keys).agg(
            parlay_tickets=("game_a","size"),
            parlay_staked=("stake","sum"),parlay_net=("profit","sum"),
            parlay_wins=("both_won","sum"),
        ).reset_index()
        base=base.merge(add,on=keys,how="left",validate="one_to_one")
    for name in ["parlay_tickets","parlay_staked","parlay_net","parlay_wins"]:
        if name not in base:
            base[name]=0
        base[name]=base[name].fillna(0)
    base["combined_staked"]=base.straight_staked+base.parlay_staked
    base["combined_net"]=base.straight_net+base.parlay_net
    base["straight_roi_pct"]=100*base.straight_net/base.straight_staked
    base["parlay_roi_pct"]=np.where(base.parlay_staked>0,
        100*base.parlay_net/base.parlay_staked,np.nan)
    base["combined_roi_pct"]=100*base.combined_net/base.combined_staked
    return base.sort_values(keys).reset_index(drop=True)


def run(winners, margins, *, expected_seasons=EVALUATED_SEASONS):
    required = {"game_id","season","model","prediction"}
    if not required.issubset(winners):
        raise ValueError(f"Missing winner columns: {sorted(required-set(winners))}")
    years = set(winners.season.unique())
    if not {2014,2015}.issubset(years):
        raise ValueError("Calibration warmup MUST include 2014 and 2015 OOS predictions")
    eligible_margin = set(margins.season.unique())
    if not {2014,2015}.issubset(eligible_margin):
        raise ValueError("Spread margin residual history MUST include 2014 and 2015")
    universe = build_candidates(winners,margins,first=2016,calibration_first=2016)
    singles = make_singles(universe)
    parlays, reviewed = parlay_from_rejected(universe,singles)
    years_selected = set(singles.season.unique())
    if not set(expected_seasons).issubset(years_selected):
        raise ValueError(f"Missing portfolio seasons: {sorted(set(expected_seasons)-years_selected)}")
    if years_selected-set(expected_seasons):
        raise ValueError("Unexpected evaluation years in portfolio")
    # Exact integer-cent full spending each week, no additional bets disguised as
    # straight wagers. Parlay sleeve is optional/extra.
    staked_by_week = singles.groupby(["policy","season","week"]).stake.sum()
    if not np.allclose(staked_by_week.values,STRAIGHT_BUDGET,atol=.005):
        raise AssertionError("Not all observed weekly $100 straight budgets used")
    if singles.duplicated(["policy","game_id"]).any():
        raise AssertionError("Multiple straights on a game")
    if len(parlays) and ((parlays.stake<0).any() or (parlays.stake>PARLAY_MAX_BUDGET).any()):
        raise AssertionError("Invalid parlay sleeve stake")
    rows = []
    for policy in sorted(universe.policy.unique()):
        ss = singles.loc[singles.policy.eq(policy)]
        ps = parlays.loc[parlays.policy.eq(policy)] if len(parlays) else parlays
        for season in expected_seasons:
            rows.append(dict(policy=policy,**_summary(ss,ps,season)))
    annual = pd.DataFrame(rows)
    totals=[]
    for policy in sorted(universe.policy.unique()):
        ss=singles.loc[singles.policy.eq(policy)]
        ps=parlays.loc[parlays.policy.eq(policy)] if len(parlays) else parlays
        totals.append(dict(policy=policy,**_summary(ss,ps)))
    comparisons=pd.DataFrame(totals)
    details={
        "ten_year_span":"2016-2025",
        "warmup_forecast_years":[2014,2015],
        "winner_calibration":"Train each season using only earlier seasons' out-of-sample winner forecasts.",
        "margin_model":"2014-2025 ridge walk-forward on earlier weeks; cover residual distribution from earlier seasons only.",
        "weekly_straight_budget":STRAIGHT_BUDGET,
        "bet_limit":"None (one straight market per game)",
        "optional_parlay":"At most one two-leg cross-game rejected-ML parlay from other games per week",
        "parlay_stake":"Extra $2-15 when qualifying; $0 otherwise; not drawn from straight-bet $100",
        "parlay_odds":"Action Network-style decimal compounding of historic individual-leg American prices.",
        "limitations":[
            "Recorded historical NFL odds have no verified simultaneous bookmaker quote/decision timestamps.",
            "The 2014-15 warmup avoids training on the 2016 test outcomes but does not make 2016-25 truly untouched: policy was conceived after historical review.",
            "Parlay leg probabilities are multiplied under independence, which may fail due to game correlations.",
            "Winning parlays with negative-value legs do not prove a positive future expected return.",
            "Betting $100 every week may force poor-value selections; no guaranteed positive ROI.",
            "A book may restrict parlay legs or not offer both historical prices simultaneously.",
        ],
        "side_by_side_overall":comparisons.to_dict("records"),
    }
    return details,annual,comparisons,singles,parlays,reviewed,universe


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--winner-predictions",required=True)
    p.add_argument("--warmup-predictions",required=True)
    p.add_argument("--output",default="results/ten_year_ml_spread_parlays")
    p.add_argument("--margin-predictions",default=None,
        help="Optional 2014-25 margin CSV; otherwise refit from 2013-25 NFL schedules.")
    a=p.parse_args()
    primary=pd.read_csv(a.winner_predictions)
    warm=pd.read_csv(a.warmup_predictions)
    all_winners=pd.concat([warm,primary],ignore_index=True)
    if all_winners.duplicated(["game_id","model"]).any():
        raise ValueError("Overlapping calibration warmup and normal replay rows")
    if a.margin_predictions:
        margins=pd.read_csv(a.margin_predictions)
    else:
        # Build a new chronological margin replay beginning 2014, not 2019.
        schedules=get_data(None,list(range(2013,2026)))
        margins=margin_replay(schedules,first=2014,last=2025)
    report,annual,overall,singles,parlays,reviewed,universe=run(all_winners,margins)
    dst=Path(a.output);dst.mkdir(parents=True,exist_ok=True)
    (dst/"report.json").write_text(json.dumps(report,indent=2))
    annual.to_csv(dst/"annual_comparison.csv",index=False)
    weekly_comparison(singles,parlays).to_csv(dst/"weekly_comparison.csv",index=False)
    overall.to_csv(dst/"ten_year_totals.csv",index=False)
    singles.to_csv(dst/"straight_bets.csv",index=False)
    if not parlays.empty:
        parlays.to_csv(dst/"calculated_cross_game_parlays.csv",index=False)
    if not reviewed.empty:
        reviewed.to_csv(dst/"parlay_candidates.csv",index=False)
    universe.to_csv(dst/"all_candidate_markets.csv",index=False)
    margins.to_csv(dst/"margin_predictions_with_warmup.csv",index=False)
    print("TEN-YEAR 2016-2025 COMPARISON")
    print(overall.to_string(index=False))
    print("SEASON-BY-SEASON")
    print(annual.to_string(index=False))
    print(json.dumps(report["limitations"],indent=2))


if __name__=="__main__":
    main()
