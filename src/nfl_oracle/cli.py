"""NFL Oracle command line: historical evaluation and unified V7 market review.

The 'shortlist' command now uses one ML/ATS straight/parlay probability
engine. The old moneyline-only bankroll/parlay shortcut is not invoked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .core import backtest


def get_data(path: str | None, seasons: list[int]) -> pd.DataFrame:
    if path:
        df=pd.read_csv(path)
    else:
        import nflreadpy as nfl
        df=nfl.load_schedules(seasons).to_pandas()
    if "game_type" in df:
        df=df.loc[df.game_type.isin(["REG"])].copy()
    return df


def main():
    ap=argparse.ArgumentParser(description="NFL Oracle: OOS tests or V7 straight and parlay research.")
    ap.add_argument("command",choices=("backtest","shortlist"))
    ap.add_argument("--schedule-csv",default=None)
    ap.add_argument("--test-year",type=int,default=2025)
    ap.add_argument("--seasons",nargs="+",type=int,default=list(range(2013,2027)))
    ap.add_argument("--as-of",help="Timezone-aware ISO timestamp for strictly pre-kickoff quotes")
    source=ap.add_mutually_exclusive_group()
    source.add_argument("--odds-csv",help="Current same-book ML+ATS quote CSV")
    source.add_argument("--live-odds",action="store_true")
    ap.add_argument("--prior-margins",help="Historical OOS score-margin predictions")
    ap.add_argument("--calibration-csv",help="Historical OOS matched football/margin calibration")
    ap.add_argument("--liked-picks",help="CSV game_id,side,kind; display preference only")
    # Retain the option for older command invocation, but never allocate stakes.
    ap.add_argument("--parlay-bankroll",type=float,default=None,
        help="Deprecated; ignored by V7 because selection is percentages-only")
    ap.add_argument("--output",default="results")
    args=ap.parse_args()

    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.command=="backtest":
        data=get_data(args.schedule_csv,args.seasons)
        report,rows=backtest(data,test_year=args.test_year)
        (output/"backtest_report.json").write_text(json.dumps(report,indent=2))
        rows.to_csv(output/"backtest_predictions.csv",index=False)
        print(json.dumps(report,indent=2))
        return

    if not args.as_of or not (bool(args.odds_csv)^bool(args.live_odds)):
        ap.error("shortlist requires --as-of and exactly one of --odds-csv or --live-odds")
    if not args.prior_margins:
        ap.error("V7 shortlist needs --prior-margins to estimate ATS distributions")
    from scripts.parlay_live_mixed import (
        parse_asof,build_preview,fetch_live_quotes)
    from scripts.parlay_mixed_market_v5 import load_policy
    from scripts.parlay_conviction_v6 import load_likes
    asof=parse_asof(args.as_of)
    schedules=get_data(args.schedule_csv,args.seasons)
    offers=(fetch_live_quotes(schedules) if args.live_odds else pd.read_csv(args.odds_csv))
    _,old_policy=load_policy()  # Compatibility input; V7 ignores V5 thresholds.
    likes=load_likes(args.liked_picks)
    calibration=(pd.read_csv(args.calibration_csv) if args.calibration_csv else None)
    straight,parlays,weekly,report=build_preview(
        schedules,offers,pd.read_csv(args.prior_margins),asof,old_policy,
        strategy="v7",liked=likes,calibration_history=calibration)
    straight.to_csv(output/"straight_picks_all_four_sides.csv",index=False)
    parlays.to_csv(output/"all_parlay_combinations.csv",index=False)
    weekly.to_csv(output/"weekly_selection_funnel.csv",index=False)
    if len(parlays):
        parlays.loc[parlays.featured_rank.notna()].sort_values(
            ["bookmaker","week","featured_rank"]).to_csv(
            output/"diversified_parlay_board.csv",index=False)
        parlays.sort_values(["bookmaker","week","preferred_rank"]).to_csv(
            output/"confidence_and_likes_board.csv",index=False)
    if len(straight) and "liked" in straight:
        straight.loc[straight.liked].to_csv(output/"liked_straights.csv",index=False)
    report["legacy_parlay_bankroll_ignored"]=args.parlay_bankroll is not None
    report["shortlist_engine"]="V7 unified ML/ATS percentage-only"
    (output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
