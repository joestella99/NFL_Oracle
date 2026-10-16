"""Ten complete NFL seasons (2016-2025): V5 cross-market parlay backtest.

Strictly walk-forward winner forecasts and score-margin forecasts. For a
season Y, ATS cover probabilities use only *out-of-sample* margin errors
from seasons < Y, never errors from the tested season. Includes 2014/15
margin warmup so 2016 can be evaluated; no hindsight selection.

All combinations with two to four legs are evaluated by exact V5
analyze_game/ticket functions but NOT kept in RAM. All qualifying tickets
and the top 20 including rejected tickets per week are exported. Reports
one identical notional unit per qualifying ticket, plus a top-one-per-week
audit. NO bankroll caps, dollars, actual wagering, or stake optimization.

The old ten-year seasons have already been inspected for model development,
and NFLverse pricing does not verify contemporaneous same-book execution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from heapq import heappush, heapreplace
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_oracle.cli import get_data
from scripts.margin_model_comparison import margin_replay
from scripts.parlay_2026_review import (
    load_injury_source, prepare_personnel_model, score_card,
)
from scripts.personnel_asof import from_nflreadpy
from scripts.parlay_mixed_market_v5 import (
    VARIANTS, analyze_game, load_policy, ticket,
)


def classify(t:dict)->str:
    if t["ats_legs"]==0:return "ML_only"
    if t["ml_legs"]==0:return "ATS_only"
    return "Mixed_ML_ATS"


def calibration_baseline(forecasts: pd.DataFrame) -> pd.DataFrame:
    """2014/15 warmup, 2016..2025 retrospective. Never train in target year."""
    expected=set(range(2014,2026))
    available=set(forecasts.season.astype(int).unique())
    if expected-available:
        raise ValueError(f"Missing year(s) in historical predictions: {sorted(expected-available)}")
    out=forecasts.loc[forecasts.season.between(2016,2025)].copy()
    if out.game_id.duplicated().any():
        raise ValueError("Duplicate historical game")
    return out


def margin_history(schedules: pd.DataFrame, recent:pd.DataFrame)->pd.DataFrame:
    """2014-2018 OOS warmup + identical existing 2019-2025 margin replay."""
    prior=margin_replay(schedules,first=2014,last=2018)
    if not recent.season.between(2019,2025).all():
        raise ValueError("Expected 2019-2025 existing margin forecasts")
    all_margin=pd.concat([prior,recent],ignore_index=True)
    if all_margin.game_id.duplicated().any():
        raise ValueError("Duplicate 2014-2025 margin prediction")
    assert set(range(2014,2026)).issubset(set(all_margin.season.unique()))
    return all_margin


def _blank()->dict:
    return dict(examined=0,passed=0,wins=0,losses=0,push_reduced=0,
                sum_unit_return=0.,raw_ev_sum=0.,cons_ev_sum=0.,
                qualified_joint_prob_sum=0.)


def simulate(
    forecasts:pd.DataFrame, margins:pd.DataFrame, policy,
    variants:tuple[str,...]=VARIANTS
)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Per-week no-lookahead simulation. No future results used in selection."""
    if not forecasts.season.between(2016,2025).all():
        raise ValueError("Forecast simulation must only include 2016-2025")
    if not set(variants).issubset(forecasts):
        raise ValueError("Missing V5 prediction variants")
    if forecasts.game_id.duplicated().any() or margins.game_id.duplicated().any():
        raise ValueError("Duplicate game data")
    needed=["game_id","season","week","spread_line","home_spread_odds",
            "away_spread_odds","margin_prediction","true_margin"]
    if not set(needed).issubset(margins):
        raise ValueError("Missing historical ATS market data")
    merged=forecasts.merge(margins[needed],on=["game_id","season","week"],
                            validate="one_to_one",how="inner")
    if merged.empty:raise ValueError("No synchronized games")
    selected=[];week_rows=[];near=[];game_legs=[]
    for season in range(2016,2026):
        prior=margins.loc[margins.season.lt(season)]
        if prior.empty or int(prior.season.max())>=season:
            raise AssertionError("Leakage: residual training must precede test season")
        errors=(prior.true_margin-prior.margin_prediction).to_numpy(float)
        errors=errors[np.isfinite(errors)]
        if len(errors)<200:raise ValueError(f"No reliable earlier-season error pool for {season}")
        year=merged.loc[merged.season.eq(season)]
        for week, slate in year.groupby("week",sort=True):
            for variant in variants:
                picks={}
                for game in slate.itertuples(index=False):
                    legs=analyze_game(game,errors,variant,policy)
                    for leg in legs:
                        game_legs.append({
                            "season":int(season),"week":int(week),"model":variant,
                            "game_id":leg["game_id"],"kind":leg["kind"],
                            "side":leg["side"],"win_probability":leg["win_probability"],
                            "push_probability":leg["push_probability"],
                            "conservative_win_probability":leg["conservative_win_probability"],
                            "decimal_odds":leg["odds_decimal"],
                            "raw_leg_ev":leg["raw_leg_ev"],
                            "conservative_leg_ev":leg["conservative_leg_ev"],
                            "candidate_selected":leg["candidate_selected"],
                            "selection_reason":leg["selection_reason"],
                            "won":leg["won"],"pushed":leg["pushed"]})
                    chosen=[leg for leg in legs if leg["candidate_selected"]]
                    if chosen:picks[str(game.game_id)]=chosen
                counts={mix:_blank() for mix in ("all","ML_only","Mixed_ML_ATS","ATS_only")}
                rejected=dict()
                best20=[]
                best_pass=None
                for length in range(2,min(policy.max_legs,len(picks))+1):
                    for ids in combinations(sorted(picks),length):
                        for choices in product(*(picks[k] for k in ids)):
                            t=ticket(choices,policy,variant)
                            mix=classify(t)
                            n_status=t["status"]
                            rejected[n_status]=rejected.get(n_status,0)+1
                            for stat in (counts["all"],counts[mix]):
                                stat["examined"]+=1
                            kval=(t["conservative_ev"],t["selections"])
                            if len(best20)<20:
                                heappush(best20,(kval,t))
                            elif kval>best20[0][0]:
                                heapreplace(best20,(kval,t))
                            if n_status!="QUALIFIES_RESEARCH":continue
                            selected.append(t)
                            for stat in (counts["all"],counts[mix]):
                                stat["passed"]+=1
                                stat["wins"]+=int(t["true_unit_return"]>0)
                                stat["losses"]+=int(t["true_unit_return"]<0)
                                stat["push_reduced"]+=int(t["contains_push"])
                                stat["sum_unit_return"]+=t["true_unit_return"]
                                stat["raw_ev_sum"]+=t["estimated_ev"]
                                stat["cons_ev_sum"]+=t["conservative_ev"]
                                stat["qualified_joint_prob_sum"]+=t["conservative_all_win_probability"]
                            if best_pass is None or (
                                t["conservative_ev"],t["estimated_all_win_probability"]
                            ) > (
                                best_pass["conservative_ev"],best_pass["estimated_all_win_probability"]
                            ):
                                best_pass=t
                for rank,(_,t) in enumerate(sorted(best20,reverse=True),1):
                    near.append(dict(**t,rank_by_conservative_ev=rank))
                row=dict(season=int(season),week=int(week),model=variant,
                    games=int(len(slate)),pre_year_residual_games=int(len(errors)),
                    last_residual_season=int(prior.season.max()),
                    candidate_games=int(len(picks)),
                    rejected_win_prob=int(rejected.get("WIN_PROBABILITY_BELOW_THRESHOLD",0)),
                    rejected_conservative_ev=int(rejected.get("CONSERVATIVE_EV_BELOW_THRESHOLD",0)),
                    top_one_selected=best_pass is not None,
                    top_one_return=(best_pass["true_unit_return"] if best_pass else 0.),
                    top_one_ev=(best_pass["conservative_ev"] if best_pass else None),
                    top_one_selections=(best_pass["selections"] if best_pass else None))
                for key,stat in counts.items():
                    suf={"all":"","ML_only":"_ml","Mixed_ML_ATS":"_mixed",
                         "ATS_only":"_ats"}[key]
                    for k,v in stat.items():row[k+suf]=v
                week_rows.append(row)
    return (pd.DataFrame(week_rows),pd.DataFrame(selected),
            pd.DataFrame(near),pd.DataFrame(game_legs))


def _roi(total_return:float,count:int)->float|None:
    return float(total_return/count) if count else None


def _resampled_interval(sub:pd.DataFrame,seed=20261016,draws=5000)->dict:
    """Year-block bootstrap; rough robustness measure, not prospective odds."""
    annual=sub.groupby("season",as_index=False).agg(n=("passed","sum"),
            unit_return=("sum_unit_return","sum"))
    if not len(annual):return {"p05":None,"p95":None}
    gen=np.random.default_rng(seed)
    ids=gen.integers(0,len(annual),size=(draws,len(annual)))
    wins=annual.unit_return.to_numpy(float)[ids].sum(axis=1)
    counts=annual.n.to_numpy(int)[ids].sum(axis=1)
    fractions=np.divide(wins,counts,out=np.full_like(wins,np.nan),where=counts>0)
    fractions=fractions[np.isfinite(fractions)]
    if not len(fractions):return {"p05":None,"p95":None}
    return {"p05":float(np.percentile(fractions,5)),
            "p95":float(np.percentile(fractions,95))}


def report_results(weekly:pd.DataFrame, tickets:pd.DataFrame,legs:pd.DataFrame,
                   policy_id:str,policy,forecast:pd.DataFrame,
                   sources:list[dict])->tuple[dict,pd.DataFrame,pd.DataFrame]:
    annual=weekly.groupby(["season","model"],as_index=False).agg(
        weeks=("week","nunique"),games=("games","sum"),
        examined=("examined","sum"),passed=("passed","sum"),
        wins=("wins","sum"),losses=("losses","sum"),
        sum_unit_return=("sum_unit_return","sum"),
        passed_ml=("passed_ml","sum"),passed_mixed=("passed_mixed","sum"),
        passed_ats=("passed_ats","sum"),top_one=("top_one_selected","sum"),
        top_one_return=("top_one_return","sum"))
    annual["equal_unit_roi"]=annual.apply(lambda r:_roi(r.sum_unit_return,int(r.passed)),axis=1)
    annual["top_one_per_week_roi"]=annual.apply(
        lambda r:_roi(r.top_one_return,int(r.top_one)),axis=1)
    total=[]
    for model,sub in weekly.groupby("model"):
        row=dict(model=model,seasons=10,weeks=int(len(sub)),games=int(sub.games.sum()),
            combinations_tested=int(sub.examined.sum()),
            qualifying_tickets=int(sub.passed.sum()),
            wins=int(sub.wins.sum()),losses=int(sub.losses.sum()),
            weeks_with_qualifiers=int((sub.passed>0).sum()),
            qualifying_ml_only=int(sub.passed_ml.sum()),
            qualifying_mixed=int(sub.passed_mixed.sum()),
            qualifying_ats_only=int(sub.passed_ats.sum()),
            equal_unit_net=float(sub.sum_unit_return.sum()),
            equal_unit_roi=_roi(sub.sum_unit_return.sum(),int(sub.passed.sum())),
            top_one_per_week_tickets=int(sub.top_one_selected.sum()),
            top_one_per_week_unit_net=float(sub.top_one_return.sum()),
            top_one_per_week_roi=_roi(sub.top_one_return.sum(),int(sub.top_one_selected.sum())),
            negative_seasons=int((annual.loc[annual.model.eq(model),"sum_unit_return"]<0).sum()),
            bootstrap_roi_year_block_p05_p95=_resampled_interval(sub))
        if len(tickets):
            t=tickets.loc[tickets.model.eq(model)]
            row["qualifying_by_length"]={str(k):int(v) for k,v in t.legs.value_counts().items()}
        total.append(row)
    overall=dict(policy_id=policy_id,
        years="2016-2025",thresholds=asdict(policy),rows=total,
        sourced_injury_counts=sources,
        year_block_bootstrap_description="Resample 10 NFL seasons with replacement 5000 times; 5th/95th percentiles of same-sample equal-ticket-unit ROI. Not a prospective success probability.",
        all_tickets_same_notional_unit=True,no_stakes_or_bankroll_constraints=True,
        selection="All V5 model-favored ML and best EV ATS choices; 2..4 legs, distinct games.",
        training="Winner model fitted on prior-year out-of-sample forecasts; margin model fitted each week on previously completed games; historical ATS cover residuals restricted to seasons BEFORE evaluation season.",
        caveats=[
            "Unverified historical NFLverse bookmaker quotes and spreads were mathematically compounded; profitable theoretical payouts may never have been available at one book and one time.",
            "2016-2025 outcomes were examined extensively to design the V5 rules. This is IN-SAMPLE policy research, not prospective proof of positive expected value.",
            "All qualifying tickets overlap games and cannot be treated as independent observations.",
            "Theoretical same-unit ROI across ALL tickets is not a realistically executable betting portfolio: wager acceptance, void/parlay rules, bookmaker limits, and game overlap are ignored.",
            "Injury inputs are recorded only where last_modified is before kickoff; missing means unknown, not healthy.",
            "Market-only no-vig win probabilities do NOT show independent market-beating win-value skill; football variants historically underperformed market Brier.",
            "No 2026 results enter this ten-season report; 2026 held-out check is separate.",
        ])
    length_and_type=(tickets.groupby(["model","legs","market_mix"],as_index=False)
            .agg(qualifying=("status","size"),total_unit_return=("true_unit_return","sum"))
            if len(tickets) else pd.DataFrame())
    if len(length_and_type):
        length_and_type["equal_unit_roi"]=(
            length_and_type.total_unit_return/length_and_type.qualifying)
    return overall,annual,length_and_type


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--warmup",required=True)
    ap.add_argument("--recent-margins",required=True)
    ap.add_argument("--output",default="results/parlay_v5_ten_year")
    ap.add_argument("--schedules",help="Optional historical schedules CSV")
    args=ap.parse_args()
    policy_id,policy=load_policy()
    schedules=get_data(args.schedules,list(range(2013,2026)))
    raw=pd.concat([pd.read_csv(args.warmup),
                   pd.read_csv(args.predictions)],ignore_index=True)
    import nflreadpy as nfl
    injuries,sources=load_injury_source(nfl,[2023,2024,2025])
    personnel,_=from_nflreadpy(schedules.loc[schedules.season.ge(2023)],injuries)
    forecast=calibration_baseline(prepare_personnel_model(raw,personnel))
    margins=margin_history(schedules,pd.read_csv(args.recent_margins))
    weekly,qualified,top,legs=simulate(forecast,margins,policy)
    report,annual,types=report_results(weekly,qualified,legs,policy_id,
                                       policy,forecast,sources)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    weekly.to_csv(output/"weekly.csv",index=False)
    annual.to_csv(output/"annual.csv",index=False)
    types.to_csv(output/"by_parlay_length_and_market.csv",index=False)
    qualified.to_csv(output/"all_qualifying_tickets.csv",index=False)
    top.to_csv(output/"top20_each_week_including_rejected.csv",index=False)
    legs.to_csv(output/"all_single_market_audits.csv",index=False)
    print("V5 TEN-YEAR MONEYLINE + ATS PARLAY RESULTS",flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print("BY YEAR\n"+annual.to_string(index=False),flush=True)
    print("BY LEG COUNT/MARKET TYPE\n"+types.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
