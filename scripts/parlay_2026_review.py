"""NFL Oracle percentage-only 2026 parlay review with historical as-of injuries.

Replays only completed/untied games. Generates point-in-time 2026 model
forecasts with no 2026 outcomes used for training (winner variants use strictly
earlier weeks). Personnel recalibration uses strictly earlier SEASONS.
No dollar budget, bankroll sizing, or bet execution. Historical odds are
NOT verified pregame executable quotes. ESPN latest API is NOT backfilled.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass,asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_oracle.cli import get_data
from scripts.backtest_ten_years import compare
from scripts.parlay_portfolio_v2 import (
    prepare_prediction_features,feature_matrix,candidate_legs,enumerate_tickets,
)
from scripts.personnel_asof import from_nflreadpy,source_summary,FIELDS

DEFAULT_POLICY_FILE=Path(__file__).resolve().parents[1]/"config"/"parlay_percentage_policy.json"


@dataclass(frozen=True)
class PercentageRule:
    """No amount, stake, bankroll, or number of parlays in these standards."""
    min_leg_probability: float=.60
    min_joint_probability: float=.15
    min_conservative_ev: float=.02
    base_probability_haircut: float=.02
    disagreement_haircut_weight: float=.20
    per_leg_price_haircut: float=.01
    require_positive_leg_ev: bool=False

    def validate(self):
        for key,val in asdict(self).items():
            if key=="require_positive_leg_ev":
                if not isinstance(val,bool):raise ValueError("Invalid leg flag")
            elif not 0<=val<1: raise ValueError("Invalid "+key)

def load_policy(path:Path=DEFAULT_POLICY_FILE) -> tuple[str,PercentageRule]:
    raw=json.loads(Path(path).read_text())
    if raw.get("status")!="official_percentage_only_parlay_policy":
        raise ValueError("Refusing nonofficial percentage policy")
    rule=PercentageRule(**raw["thresholds"])
    rule.validate()
    if rule.min_conservative_ev!=.02 or rule.per_leg_price_haircut!=.01:
        raise ValueError("Official parlay +2% combined EV and 1% per-leg price stress are required")
    if rule.require_positive_leg_ev:
        raise ValueError("Do not reinstate individual-leg veto")
    return raw["policy_id"],rule

def _predict(frame:pd.DataFrame, additional:bool) -> np.ndarray:
    a=feature_matrix(frame)
    if not additional:return a
    return np.column_stack([a,frame[list(FIELDS)].to_numpy(float)])

def prepare_personnel_model(all_forecasts:pd.DataFrame,personnel:pd.DataFrame) -> pd.DataFrame:
    frame=prepare_prediction_features(all_forecasts).merge(
        personnel,on="game_id",how="left",validate="one_to_one")
    for name in FIELDS:
        frame[name]=pd.to_numeric(frame[name],errors="coerce").fillna(0.)
    frame["football_base_prob"]=np.nan
    frame["football_injury_prob"]=np.nan
    for year in sorted(frame.season.unique()):
        if year<2016:continue
        train=frame.loc[frame.season.lt(year)]
        test=frame.loc[frame.season.eq(year)]
        if len(train)<100 or train.season.nunique()<2 or test.empty:continue
        for name,enhanced in [("football_base_prob",False),("football_injury_prob",True)]:
            estimator=make_pipeline(StandardScaler(),
                LogisticRegression(C=.05,max_iter=1000,random_state=20261016))
            estimator.fit(_predict(train,enhanced),train.home_win.astype(int))
            frame.loc[test.index,name]=estimator.predict_proba(_predict(test,enhanced))[:,1]
    return frame


def choose_forecast(frame:pd.DataFrame) -> tuple[str,pd.DataFrame]:
    """Choose a 2026 model using only 2023-25 outcomes, NOT 2026 winners.

    Compare to market too. Market calibration wins if no football estimator
    shows lower past Brier. This prevents relabeling a weak model 'trusted.'
    """
    history=frame.loc[frame.season.between(2023,2025)]
    variants=["market_home_prob","football_base_prob","football_injury_prob"]
    measures={}
    for name in variants:
        subset=history.loc[history[name].notna()]
        measures[name]=float(np.mean((subset[name]-subset.home_win)**2))
    winner=min(variants,key=lambda n:measures[n])
    return winner,pd.DataFrame([{"variant":k,"brier_2023_2025":v,
        "comparison_games":int(history[k].notna().sum())} for k,v in measures.items()])


def score_card(frame:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for year,g in frame.groupby("season"):
        if year<2023:continue
        for name in ["market_home_prob","football_base_prob","football_injury_prob"]:
            s=g.loc[g[name].notna()]
            if len(s):
                rows.append(dict(season=int(year),variant=name,games=len(s),
                    brier=float(np.mean((s[name]-s.home_win)**2)),
                    correct=int(((s[name]>=.5)==s.home_win).sum()),
                    accuracy=float(np.mean((s[name]>=.5)==s.home_win)),
                    qb_reports_both=int(s.qb_report_coverage.sum()),
                    ol_reports_both=int(s.ol_report_coverage.sum())))
    return pd.DataFrame(rows)


def percentage_shortlist(frame:pd.DataFrame,rule:PercentageRule,
                         chosen_variant:str) -> tuple[pd.DataFrame,pd.DataFrame]:
    rule.validate()
    audited=[];weekly=[]
    outcomes=frame.set_index("game_id").home_win.astype(bool).to_dict()
    for (season,week),slate in frame.loc[frame.season.eq(2026)].groupby(
            ["season","week"],sort=True):
        games=slate.copy()
        games["football_home_prob"]=games[chosen_variant]
        legs=candidate_legs(games,rule)
        tickets=enumerate_tickets(legs,rule)
        if len(tickets):
            for t in tickets.itertuples(index=False):
                won=all(outcomes[gid] if side=="home" else not outcomes[gid]
                        for gid,side in zip(t.game_ids,t.sides))
                audited.append(dict(season=int(season),week=int(week),
                    games=" | ".join(t.game_ids), sides=" | ".join(t.sides),
                    legs=int(t.legs),joint_model_probability=float(t.model_joint_probability),
                    joint_conservative_probability=float(t.risk_adjusted_joint_probability),
                    combined_ev=float(t.estimated_ev),conservative_ev=float(t.conservative_ev),
                    weak_leg_count=int(t.weak_leg_count),status=t.status,
                    won=bool(won), historical_price_proxy=True,
                    checked_against_bookmaker_execution=False))
        weekly.append(dict(season=int(season),week=int(week),
            completed_games=len(slate),eligible_legs=len(legs),
            evaluated_combinations=len(tickets),
            passed_criteria=int(tickets.status.eq("ELIGIBLE_RESEARCH").sum()) if len(tickets) else 0))
    return pd.DataFrame(audited),pd.DataFrame(weekly)


def load_injury_source(nfl, seasons:list[int]) -> tuple[pd.DataFrame,list[dict]]:
    sources=[];tables=[]
    for year in seasons:
        try:
            df=nfl.load_injuries(year).to_pandas()
            sources.append(dict(season=year,available=True,records=len(df)))
            if len(df): tables.append(df)
        except Exception as exc:
            sources.append(dict(season=year,available=False,records=0,
                                error_type=type(exc).__name__,
                                note="Not available; never treat missing data as healthy"))
    return (pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()),sources


def run(*,prior_predictions:pd.DataFrame,warmup:pd.DataFrame,
        schedules:pd.DataFrame,pbp:pd.DataFrame,injuries:pd.DataFrame,
        injury_sources:list[dict],rule:PercentageRule=PercentageRule()):
    if schedules.season.max()<2026:raise ValueError("Missing 2026 schedule")
    recent,raw=compare(schedules,pbp,first=2025,last=2026)
    forecast2026=raw.loc[raw.season.eq(2026)].copy()
    if forecast2026.empty:
        raise ValueError("No completed 2026 games with valid pregame odds/PBP features")
    combined=pd.concat([warmup,prior_predictions,forecast2026],ignore_index=True)
    if combined.duplicated(["game_id","model"]).any():
        raise ValueError("Duplicate historical or 2026 model forecast")
    # Only 2023-forward injury source is available. Pre-2023 unknown is explicit.
    p,g=from_nflreadpy(schedules.loc[schedules.season.ge(2023)],injuries)
    probs=prepare_personnel_model(combined,p)
    scores=score_card(probs)
    choice,train_scores=choose_forecast(probs)
    picks,weekly=percentage_shortlist(probs,rule,choice)
    year2026=probs.loc[probs.season.eq(2026)]
    assert len(year2026)>0
    assert not year2026[["football_base_prob","football_injury_prob"]].isna().any().any()
    report={
        "status":"2026 completed-games held-out evaluation, no dollar allocation",
        "selected_2026_probability_model":choice,
        "selected_using":"2023-2025 completed results only; no 2026 winners for selector",
        "game_count_2026":int(len(year2026)),
        "weeks_completed_or_partially_completed":sorted(year2026.week.astype(int).unique().tolist()),
        "injury_sources":injury_sources,
        "injury_coverage":source_summary(p.loc[p.game_id.isin(year2026.game_id)],g.loc[g.game_id.isin(year2026.game_id)]),
        "risk_standards":asdict(rule),
        "ticket_candidates_2026":int(len(picks)),
        "tickets_pass_2026":int(picks.status.eq("ELIGIBLE_RESEARCH").sum()) if len(picks) else 0,
        "dollar_bets":None,
        "real_wagering":False,
        "limitations":[
            "Only completed 2026 games; no upcoming game outcomes used as current-season test labels.",
            "Some observed 2026 moneylines in nflverse may be retrospectively recorded, not timestamped sportsbook offers.",
            "nflreadpy injury records require source last-modified < kickoff; records overwritten later are discarded.",
            "Historical injured QB is not proof of which starting QB was unavailable.",
            "ESPN live injury or depth chart views cannot reconstruct historical snapshots.",
            "Training-year predictions and feature design were researched on 2016-2025 already; 2026 analysis is a retrospective check, not prespecified untouched betting edge.",
            "No claim about current future-game parlay availability is made.",
        ]
    }
    return report,probs,scores,train_scores,picks,weekly,p,g


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--warmup",required=True)
    ap.add_argument("--output",default="results/parlay_2026_percentages")
    ap.add_argument("--first-data-season",type=int,default=2013)
    ap.add_argument("--policy-config",default=str(DEFAULT_POLICY_FILE))
    a=ap.parse_args()
    policy_id,rule=load_policy(Path(a.policy_config))
    import nflreadpy as nfl
    years=list(range(a.first_data_season,2027))
    schedules=get_data(None,years)
    pbp=nfl.load_pbp(years).select(
        ["game_id","posteam","defteam","epa","play_type","passer_player_id"]).to_pandas()
    injuries,sources=load_injury_source(nfl,[2023,2024,2025,2026])
    report,probs,scores,training,picks,weekly,p,audit=run(
        prior_predictions=pd.read_csv(a.predictions),
        warmup=pd.read_csv(a.warmup),
        schedules=schedules,pbp=pbp,injuries=injuries,
        injury_sources=sources,rule=rule)
    report["official_policy_id"]=policy_id
    dst=Path(a.output);dst.mkdir(parents=True,exist_ok=True)
    (dst/"report.json").write_text(json.dumps(report,indent=2))
    probs.loc[probs.season.ge(2023)].to_csv(dst/"model_probabilities.csv",index=False)
    scores.to_csv(dst/"brier_by_season.csv",index=False)
    training.to_csv(dst/"2026_model_choice_2023_2025.csv",index=False)
    picks.to_csv(dst/"2026_ticket_combinations.csv",index=False)
    weekly.to_csv(dst/"2026_weekly_funnel.csv",index=False)
    p.loc[p.game_id.isin(probs.loc[probs.season.eq(2026),"game_id"])].to_csv(
        dst/"2026_injury_features.csv",index=False)
    audit.to_csv(dst/"source_timestamp_audit.csv",index=False)
    print(json.dumps(report,indent=2),flush=True)
    print("CALIBRATION BY SEASON\n"+scores.to_string(index=False),flush=True)
    print("2026 WEEKLY FUNNEL\n"+weekly.to_string(index=False),flush=True)

if __name__=="__main__":main()
