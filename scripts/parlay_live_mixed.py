"""Upcoming NFL ML+ATS parlay watchlist from contemporaneous same-book quotes.

Unlike the 2026 completed-game research replay, this process explicitly
includes games scheduled after --as-of. No realized future outcome is loaded.
Produces all 2-4 leg combination audits (ML+ML, ML+ATS, ATS+ATS).
Two-sided moneyline and spread offers must be from the SAME sportsbook.
A quote CSV must specify as-of time; live Odds API requires ODDS_API_KEY.

Prices shown are quote snapshots, not proof the sportsbook will accept an
actual parlay. No stake, bankroll, bet automation or dollar recommendation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from nfl_oracle.cli import get_data
from nfl_oracle.core import (
    FEATURES,make_features,eligible,fit_model,no_vig_home
)
from nfl_oracle.odds import NFL_TEAMS,ALIASES
from scripts.parlay_mixed_market_v5 import build_slates,load_policy

QUOTE_COLS=("game_id","bookmaker","captured_at","home_moneyline",
            "away_moneyline","spread_line","home_spread_odds",
            "away_spread_odds")


def kickoff_utc(games:pd.DataFrame)->pd.Series:
    required={"gameday","gametime"}
    if not required.issubset(games):raise ValueError("Missing kickoff date and time")
    when=pd.to_datetime(
        games.gameday.astype(str)+" "+games.gametime.astype(str),errors="coerce")
    return when.dt.tz_localize("America/New_York",
        ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC")


def parse_asof(value:str)->pd.Timestamp:
    ts=pd.Timestamp(value)
    if ts.tz is None:raise ValueError("as-of must include time zone offset")
    return ts.tz_convert("UTC")


def extract_same_book_markets(events:list,schedule:pd.DataFrame)->pd.DataFrame:
    """Both ML and spread at one bookmaker and near the same update time."""
    sched=schedule.copy()
    sched["game_date"]=pd.to_datetime(sched.gameday).dt.date
    rows=[]
    for event in events:
        home=NFL_TEAMS.get(event.get("home_team"))
        away=NFL_TEAMS.get(event.get("away_team"))
        when=pd.to_datetime(event.get("commence_time"),utc=True,errors="coerce")
        if home is None or away is None or pd.isna(when):continue
        date=when.tz_convert("America/New_York").date()
        matched=sched.loc[
            sched.home_team.replace(ALIASES).eq(home)&
            sched.away_team.replace(ALIASES).eq(away)&
            sched.game_date.eq(date)]
        if len(matched)!=1:continue
        for book in event.get("bookmakers",[]):
            offers={x.get("key"):x for x in book.get("markets",[])}
            if "h2h" not in offers or "spreads" not in offers:continue
            ml,ats=offers["h2h"],offers["spreads"]
            m={x.get("name"):x for x in ml.get("outcomes",[])}
            s={x.get("name"):x for x in ats.get("outcomes",[])}
            home_name=event["home_team"];away_name=event["away_team"]
            if any(x not in m or x not in s for x in (home_name,away_name)):
                continue
            try:
                home_spread=float(s[home_name]["point"])
                away_spread=float(s[away_name]["point"])
                if abs(home_spread+away_spread)>.01:continue
                home_ml=float(m[home_name]["price"])
                away_ml=float(m[away_name]["price"])
                home_ats=float(s[home_name]["price"])
                away_ats=float(s[away_name]["price"])
            except (ValueError,TypeError,KeyError):continue
            t_ml=pd.to_datetime(ml.get("last_update"),utc=True,errors="coerce")
            t_ats=pd.to_datetime(ats.get("last_update"),utc=True,errors="coerce")
            if pd.isna(t_ml) or pd.isna(t_ats):continue
            if abs(t_ml-t_ats)>pd.Timedelta(minutes=10):continue
            rows.append(dict(game_id=str(matched.iloc[0].game_id),
                bookmaker=str(book.get("key") or book.get("title") or "unknown"),
                captured_at=max(t_ml,t_ats).isoformat(),
                home_moneyline=home_ml,away_moneyline=away_ml,
                # nflverse spread_line = home EXPECTED victory margin.
                spread_line=-home_spread,
                home_spread_odds=home_ats,away_spread_odds=away_ats))
    return pd.DataFrame(rows,columns=QUOTE_COLS)


def fetch_live_quotes(schedule:pd.DataFrame)->pd.DataFrame:
    key=os.getenv("ODDS_API_KEY")
    if not key:raise ValueError(
        "ODDS_API_KEY required for live ML+ATS quotes; or use --quotes-csv")
    endpoint=("https://api.the-odds-api.com/v4/sports/"
         "americanfootball_nfl/odds/?"+urlencode(
            {"apiKey":key,"regions":"us","markets":"h2h,spreads",
             "oddsFormat":"american"}))
    with urlopen(endpoint,timeout=20) as response:
        data=json.load(response)
    if not isinstance(data,list):raise ValueError("Malformed odds provider response")
    return extract_same_book_markets(data,schedule)


def check_quotes(quotes:pd.DataFrame,schedule:pd.DataFrame,
                 asof:pd.Timestamp,hours:int=12)->pd.DataFrame:
    if not set(QUOTE_COLS).issubset(quotes):
        raise ValueError("ML+ATS quote CSV requires "+", ".join(QUOTE_COLS))
    if quotes.empty:return quotes.copy()
    q=quotes.copy()
    q["captured_at"]=pd.to_datetime(q.captured_at,utc=True,errors="raise")
    q=q.loc[q.captured_at.between(asof-pd.Timedelta(hours=hours),asof)]
    q=q.loc[q.bookmaker.notna()&q.bookmaker.astype(str).str.len().gt(0)]
    sc=schedule[["game_id","kickoff_utc","season","week"]].drop_duplicates("game_id")
    q=q.merge(sc,on="game_id",how="inner",validate="many_to_one")
    q=q.loc[q.kickoff_utc.gt(asof)]
    for c in ["home_moneyline","away_moneyline","spread_line",
              "home_spread_odds","away_spread_odds"]:
        q[c]=pd.to_numeric(q[c],errors="coerce")
    q=q.dropna(subset=list(QUOTE_COLS)[3:])
    q=q.loc[q.home_moneyline.abs().ge(100)&q.away_moneyline.abs().ge(100)
         &q.home_spread_odds.abs().ge(100)&q.away_spread_odds.abs().ge(100)]
    return q.sort_values("captured_at",ascending=False).drop_duplicates(
        ["game_id","bookmaker"],keep="first")


def build_preview(schedules:pd.DataFrame,quotes:pd.DataFrame,prior_margins:pd.DataFrame,
                  asof:pd.Timestamp,policy):
    schedule=schedules.loc[schedules.game_type.eq("REG")].copy()
    schedule["kickoff_utc"]=kickoff_utc(schedule)
    cutoff=asof.tz_convert("America/New_York").tz_localize(None)
    # Do NOT allow schedule rows showing later final scores to update features.
    masked=schedule.copy()
    masked.loc[masked.kickoff_utc.ge(asof),["home_score","away_score"]]=float("nan")
    market=check_quotes(quotes,masked,asof)
    if market.empty:
        return (pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),
            {"status":"NO_VERIFIED_CURRENT_TWO_MARKET_QUOTES","games":0})
    features=make_features(masked)
    training=eligible(features,min_history=0)
    training=training.loc[training.gameday.le(cutoff)]
    upcoming=features.merge(market,on="game_id",how="inner",
        validate="one_to_many",suffixes=("_old",""))
    if upcoming.empty:
        return (pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),
            {"status":"NO_UPCOMING_MATCHING_GAMES","games":0})
    for c in ["home_moneyline","away_moneyline"]:
        # Match projection to supplied current pregame lines, not schedule line.
        upcoming[c]=upcoming[c].astype(float)
    upcoming["market_home_prob"]=upcoming.apply(
        lambda r:no_vig_home(r.home_moneyline,r.away_moneyline),axis=1)
    p=upcoming.market_home_prob.clip(1e-6,1-1e-6)
    upcoming["market_logit"]=np.log(p/(1-p))
    winner=fit_model(training,feature_cols=FEATURES)
    upcoming["pregame_form_home_prob"]=winner.predict_proba(
        upcoming[FEATURES])[:,1]
    # Margin model uses same pregame form+spread feature family used by the
    # existing walk-forward margin model, fitted only on already final games.
    cols=[x for x in FEATURES if x!="market_logit"]+["spread_line"]
    margin_train=training.merge(
        masked[["game_id","spread_line","home_score","away_score"]],
        on="game_id",how="left",validate="one_to_one")
    margin_train["actual_margin"]=margin_train.home_score-margin_train.away_score
    margin_train=margin_train.dropna(subset=["actual_margin"])
    if len(margin_train)<200:raise ValueError("Insufficient earlier games for spread model")
    estimator=make_pipeline(SimpleImputer(strategy="median"),
        StandardScaler(),Ridge(alpha=100.))
    estimator.fit(margin_train[cols],margin_train.actual_margin)
    upcoming["margin_prediction"]=estimator.predict(upcoming[cols])
    upcoming["true_margin"]=float("nan")
    upcoming["season"]=upcoming.season.astype(int)
    upcoming["week"]=upcoming.week.astype(int)
    # Compare each SAME-BOOK slate individually; never silently cross books.
    alllegs=[];alltickets=[];allweeks=[]
    for bookmaker,g in upcoming.groupby("bookmaker"):
        forecasts=g[["game_id","season","week",
            "market_home_prob","pregame_form_home_prob"]].copy()
        margins=g[["game_id","season","week","spread_line",
            "home_spread_odds","away_spread_odds","margin_prediction","true_margin"]]
        legs,tickets,weeks=build_slates(forecasts,margins,prior_margins,policy,
            variants=("market_home_prob","pregame_form_home_prob"))
        for obj in (legs,tickets,weeks):
            if len(obj):obj["bookmaker"]=bookmaker
        alllegs.append(legs);alltickets.append(tickets);allweeks.append(weeks)
    legs=pd.concat(alllegs,ignore_index=True) if alllegs else pd.DataFrame()
    tickets=pd.concat(alltickets,ignore_index=True) if alltickets else pd.DataFrame()
    weeks=pd.concat(allweeks,ignore_index=True) if allweeks else pd.DataFrame()
    report=dict(status="QUOTED_PREGAME_PERCENTAGE_REVIEW",
        asof_utc=asof.isoformat(),matched_games=int(upcoming.game_id.nunique()),
        sportsbook_count=int(upcoming.bookmaker.nunique()),
        quote_rows=int(len(upcoming)),
        reviewed_tickets=len(tickets),
        qualifying=int(tickets.status.eq("QUALIFIES_RESEARCH").sum()) if len(tickets) else 0,
        limitation=[
            "Only h2h and spread offers from the same bookmaker and within ten minutes qualify.",
            "Matched bookmaker quote last_update is not evidence of ticket acceptance.",
            "Historical margin model residuals are not prospectively calibrated on 2026 spreads.",
            "Preliminary football alternative uses pregame form rather than the full PBP/injury ensemble and is experimental.",
            "No dollar stakes, bankroll selection, or automatic betting.",
        ])
    return legs,tickets,weeks,report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--as-of",required=True,
        help="ISO8601 instant WITH timezone, e.g. 2026-10-16T15:38:00-04:00")
    source=a.add_mutually_exclusive_group(required=True)
    source.add_argument("--quotes-csv")
    source.add_argument("--live-odds",action="store_true")
    a.add_argument("--schedules-csv")
    a.add_argument("--prior-margins",required=True)
    a.add_argument("--output",default="results/parlay_pregame_mixed")
    args=a.parse_args()
    asof=parse_asof(args.as_of)
    policy_id,policy=load_policy()
    schedules=get_data(args.schedules_csv,list(range(2013,2027)))
    quotes=(fetch_live_quotes(schedules) if args.live_odds else pd.read_csv(args.quotes_csv))
    legs,tickets,weeks,report=build_preview(
        schedules,quotes,pd.read_csv(args.prior_margins),asof,policy)
    report["policy_id"]=policy_id
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    (root/"report.json").write_text(json.dumps(report,indent=2))
    legs.to_csv(root/"market_side_audit.csv",index=False)
    tickets.to_csv(root/"all_mixed_parlays.csv",index=False)
    weeks.to_csv(root/"weekly_funnel.csv",index=False)
    if len(tickets):
        tickets.sort_values(["conservative_ev","estimated_all_win_probability"],
            ascending=[False,False]).head(100).to_csv(
            root/"top_100_including_rejected.csv",index=False)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
