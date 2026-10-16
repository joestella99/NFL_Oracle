"""NFL Oracle V5: percentage-only cross-game ML, ATS, and mixed parlays.

REPLAY: 2019-25 out-of-sample margin residuals; 2026 game-by-game
out-of-sample moneyline/football forecasts. No stake or bankroll cap.
Margins predict spread-cover probabilities with a strictly earlier-season
empirical residual CDF. Exact spread pushes VOID only that leg; full ticket
loses if any other leg loses.

Audits all four sides of each game; builds 2..4-leg cross-game combinations
from the best model-favored moneyline side and the best expected-value ATS
side of each game, without a hard individual-leg EV veto. 100% of combos
are recorded, even negative estimated EV / low hit-probability candidates.

Historical lines are NOT proof of simultaneously executable offers. Past
2016-25 research is extensively reused; 2026 retrospective test is not a
prespecified live experiment. No future or unplayed games are scored.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from itertools import combinations,product
from math import prod,isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_oracle.cli import get_data
from scripts.margin_model_comparison import margin_replay
from scripts.parlay_portfolio_v2 import american_decimal

CONFIG=Path(__file__).resolve().parents[1]/"config"/"parlay_mixed_market_policy.json"
VARIANTS=("market_home_prob","football_base_prob","football_injury_prob")


@dataclass(frozen=True)
class MixedPolicy:
    min_ml_leg_probability:float=.60
    min_ats_leg_win_probability:float=.42
    min_ticket_all_win_probability:float=.15
    min_ticket_conservative_ev:float=.02
    base_probability_haircut:float=.02
    disagreement_haircut_weight:float=.20
    per_leg_price_haircut:float=.01
    max_legs:int=4

    def validate(self):
        for field,val in asdict(self).items():
            if field=="max_legs":
                if not isinstance(val,int) or not 2<=val<=4:
                    raise ValueError("max_legs must be between two and four")
            elif not 0<=val<1 or not isfinite(val):
                raise ValueError("Bad selection standard: "+field)

def load_policy(path:Path=CONFIG)->tuple[str,MixedPolicy]:
    config=json.loads(path.read_text())
    if config.get("status")!="official_mixed_market_percentage_policy":
        raise ValueError("Not official mixed-market policy")
    policy=MixedPolicy(**config["thresholds"])
    policy.validate()
    if policy.min_ticket_conservative_ev!=.02 or policy.per_leg_price_haircut!=.01:
        raise ValueError("Must preserve approved +2% combined EV / 1% leg price stress")
    return config["policy_id"],policy


def join_forecasts(forecasts:pd.DataFrame,margins:pd.DataFrame)->pd.DataFrame:
    cols=("game_id","season","week","spread_line","home_spread_odds",
          "away_spread_odds","margin_prediction","true_margin")
    if not set(cols).issubset(margins):raise ValueError("Missing margin fields")
    if margins.game_id.duplicated().any() or forecasts.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs")
    f=forecasts.copy()
    for name in VARIANTS:
        if name not in f:raise ValueError("Missing model variant "+name)
    rows=f.merge(margins[list(cols)],on=["game_id","season","week"],
                 how="inner",validate="one_to_one")
    if rows.empty:raise ValueError("No synchronized moneyline and spread games")
    return rows


def empirical_cover(predicted:float,line:float,prior_residuals:np.ndarray)->tuple[float,float,float]:
    if len(prior_residuals)<200 or not np.isfinite(prior_residuals).all():
        raise ValueError("Require >=200 past out-of-sample errors")
    if not (isfinite(predicted) and isfinite(line)):
        raise ValueError("Missing spread estimate")
    outcomes=np.rint(float(predicted)+prior_residuals)
    delta=outcomes-float(line)
    ph=float(np.mean(delta>1e-9))
    pa=float(np.mean(delta< -1e-9))
    pp=float(np.mean(np.abs(delta)<=1e-9))
    assert np.isclose(ph+pa+pp,1)
    return ph,pa,pp


def _valid_price(value)->float|None:
    try:return american_decimal(value)
    except (ValueError,TypeError,OverflowError):return None


def analyze_game(game,errors:np.ndarray,variant:str,policy:MixedPolicy)->list[dict]:
    ph=float(getattr(game,variant))
    if not isfinite(ph) or not 0<ph<1:return []
    market=float(game.market_home_prob)
    margin=float(game.true_margin) if pd.notna(game.true_margin) else float("nan")
    realized=isfinite(margin)
    cover_home,cover_away,push=empirical_cover(
        float(game.margin_prediction),float(game.spread_line),errors)
    spread_market_home_prob=None
    h_at=_valid_price(game.home_spread_odds)
    a_at=_valid_price(game.away_spread_odds)
    if h_at is not None and a_at is not None:
        home_imp=1/h_at;away_imp=1/a_at
        spread_market_home_prob=home_imp/(home_imp+away_imp)
    all_legs=[]
    for kind in ("ML","ATS"):
        for side in ("home","away"):
            is_home=side=="home"
            if kind=="ML":
                p=ph if is_home else 1-ph
                pm=market if is_home else 1-market
                q=0.
                dec=_valid_price(game.home_moneyline if is_home else game.away_moneyline)
                won=((margin>0 if is_home else margin<0) if realized else None)
                pushed=(False if realized else None)
            else:
                p=cover_home if is_home else cover_away
                q=push
                pm=(spread_market_home_prob if is_home else 1-spread_market_home_prob) if spread_market_home_prob is not None else .5
                dec=h_at if is_home else a_at
                delta=margin-float(game.spread_line)
                won=((delta>0 if is_home else delta<0) if realized else None)
                pushed=(abs(delta)<1e-9 if realized else None)
            if dec is None:continue
            conservative=max(0.,p-policy.base_probability_haircut
                -policy.disagreement_haircut_weight*abs(p-pm))
            score=p*dec+q-1
            conservative_ev=conservative*dec*(1-policy.per_leg_price_haircut)+q-1
            all_legs.append(dict(
                game_id=str(game.game_id),season=int(game.season),week=int(game.week),
                kind=kind,side=side,win_probability=float(p),push_probability=float(q),
                conservative_win_probability=float(conservative),
                market_probability=float(pm),odds_decimal=float(dec),
                raw_leg_ev=float(score),conservative_leg_ev=float(conservative_ev),
                true_margin=margin,spread_line=float(game.spread_line),
                won=bool(won),pushed=bool(pushed),
                candidate_selected=False,selection_reason="NOT_SELECTED_SIDE",
            ))
    # Select one moneyline side per game: the football model's most probable
    # winner. Any market underdog remains visible in the full audit.
    ml=[x for x in all_legs if x["kind"]=="ML"]
    if ml:
        best=max(ml,key=lambda x:(x["win_probability"],x["raw_leg_ev"]))
        if best["win_probability"]>=policy.min_ml_leg_probability:
            best["candidate_selected"]=True
            best["selection_reason"]="MODEL_FAVORED_ML"
        else:best["selection_reason"]="ML_BELOW_LEG_PROBABILITY_MINIMUM"
    # Select whichever spread side provides higher model estimated gross
    # expected return. This may be an UNDERDOG that loses ML but covers.
    ats=[x for x in all_legs if x["kind"]=="ATS"]
    if ats:
        best=max(ats,key=lambda x:(x["raw_leg_ev"],x["win_probability"]))
        if best["win_probability"]>=policy.min_ats_leg_win_probability:
            best["candidate_selected"]=True
            best["selection_reason"]="BEST_MODEL_ATS_SIDE"
        else:best["selection_reason"]="ATS_BELOW_LEG_PROBABILITY_MINIMUM"
    return all_legs


def ticket(legs:tuple[dict,...],policy:MixedPolicy,variant:str)->dict:
    if len({x["game_id"] for x in legs})!=len(legs):
        raise ValueError("Same game may not appear twice in a parlay")
    pw=prod(x["win_probability"] for x in legs)
    pc=prod(x["conservative_win_probability"] for x in legs)
    decimal=prod(x["odds_decimal"] for x in legs)
    # For independent legs: parlay gross return factor is product of each
    # leg's expected factor. Spread PUSH is a refund of just that leg.
    ev=prod(x["win_probability"]*x["odds_decimal"]+x["push_probability"]
            for x in legs)-1
    conservative_ev=prod(
        x["conservative_win_probability"]*x["odds_decimal"]*
        (1-policy.per_leg_price_haircut)+x["push_probability"]
        for x in legs)-1
    verdict=("WIN_PROBABILITY_BELOW_THRESHOLD" if pc<policy.min_ticket_all_win_probability else
             "CONSERVATIVE_EV_BELOW_THRESHOLD" if conservative_ev<policy.min_ticket_conservative_ev else
             "QUALIFIES_RESEARCH")
    # Equal-unit historical per-ticket P/L, NOT allocation or live ROI.
    known_results=all(x["won"] is not None for x in legs)
    gross_factor=(prod(1. if x["pushed"] else (x["odds_decimal"] if x["won"] else 0.)
                      for x in legs) if known_results else None)
    kinds=" + ".join(x["kind"] for x in legs)
    return dict(
        season=legs[0]["season"],week=legs[0]["week"],model=variant,
        game_ids=" | ".join(x["game_id"] for x in legs),
        selections=" | ".join(x["game_id"]+":"+x["side"]+":"+x["kind"] for x in legs),
        legs=len(legs),market_mix=kinds,
        ml_legs=sum(x["kind"]=="ML" for x in legs),
        ats_legs=sum(x["kind"]=="ATS" for x in legs),
        estimated_all_win_probability=pw,
        conservative_all_win_probability=pc,
        estimated_ev=ev,conservative_ev=conservative_ev,
        price_proxy_decimal=decimal,
        contains_push=any(x["pushed"] is True for x in legs),
        true_unit_return=(gross_factor-1 if gross_factor is not None else None),
        status=verdict,
        historical_price_proxy=True,
    )


def build_slates(forecasts:pd.DataFrame,margin_predictions:pd.DataFrame,
                 earlier_margins:pd.DataFrame,policy:MixedPolicy,
                 variants:tuple[str,...]=VARIANTS)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    games=join_forecasts(forecasts,margin_predictions)
    if games.season.max()>2026:raise ValueError("Replay restricted to 2026")
    earlier=earlier_margins.copy()
    earlier=earlier.loc[(earlier.season<games.season.min())&earlier.true_margin.notna()]
    errors=(earlier.true_margin-earlier.margin_prediction).to_numpy(float)
    errors=errors[np.isfinite(errors)]
    if len(errors)<200:raise ValueError("At least 200 previous-season margin errors required")
    leg_records=[];all_tickets=[];weeks=[]
    for variant in variants:
        for (season,week),slate in games.groupby(["season","week"],sort=True):
            choices={}
            for g in slate.itertuples(index=False):
                obs=analyze_game(g,errors,variant,policy)
                leg_records.extend({**x,"model":variant} for x in obs)
                picked=[x for x in obs if x["candidate_selected"]]
                if picked:choices[str(g.game_id)]=picked
            slate_tickets=[]
            for length in range(2,min(policy.max_legs,len(choices))+1):
                for group in combinations(sorted(choices),length):
                    for legs in product(*(choices[g] for g in group)):
                        slate_tickets.append(ticket(legs,policy,variant))
            all_tickets.extend(slate_tickets)
            passed=[t for t in slate_tickets if t["status"]=="QUALIFIES_RESEARCH"]
            weeks.append(dict(season=int(season),week=int(week),model=variant,
                completed_games=int(len(slate)),
                available_distinct_games=int(len(choices)),
                selected_game_market_legs=int(sum(len(x) for x in choices.values())),
                combinations=len(slate_tickets),qualifying=len(passed),
                qualifying_ml_only=sum(x["ats_legs"]==0 for x in passed),
                qualifying_mixed=sum(x["ml_legs"]>0 and x["ats_legs"]>0 for x in passed),
                qualifying_ats_only=sum(x["ml_legs"]==0 for x in passed)))
    return pd.DataFrame(leg_records),pd.DataFrame(all_tickets),pd.DataFrame(weeks)


def probability_diagnostics(all_legs:pd.DataFrame,tickets:pd.DataFrame)->dict:
    if tickets.empty:return {"total":0,"qualifying":0}
    research=tickets.loc[tickets.status.eq("QUALIFIES_RESEARCH")]
    return dict(
        total=len(tickets),qualifying=len(research),
        by_status={str(k):int(v) for k,v in tickets.status.value_counts().items()},
        market_mix_counts={
            "ML_only":int((tickets.ats_legs==0).sum()),
            "ML_ATS":int(((tickets.ats_legs>0)&(tickets.ml_legs>0)).sum()),
            "ATS_only":int((tickets.ml_legs==0).sum()),
        },
        week5_research_candidates=int(len(research.loc[research.week.eq(5)])),
        observed_push_legs=int(all_legs.loc[
            all_legs.kind.eq("ATS")&all_legs.pushed].shape[0]),
        note="Candidate combinations are overlapping, not independent opportunities.",
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--forecasts",required=True)
    p.add_argument("--prior-margins",required=True)
    p.add_argument("--policy-config",default=str(CONFIG))
    p.add_argument("--output",default="results/parlay_mixed_market_2026")
    p.add_argument("--schedules",help="Cached nflverse schedule CSV; else fetch nflreadpy 2013-2026")
    args=p.parse_args()
    policy_id,policy=load_policy(Path(args.policy_config))
    schedules=get_data(args.schedules,list(range(2013,2027)))
    new_margins=margin_replay(schedules,first=2026,last=2026)
    forecast=pd.read_csv(args.forecasts).loc[lambda x:x.season.eq(2026)]
    earlier=pd.read_csv(args.prior_margins)
    legs,tickets,weeks=build_slates(forecast,new_margins,earlier,policy)
    summary=dict(policy_id=policy_id,model_year=2026,standards=asdict(policy),
        evaluated_models=sorted(tickets.model.unique().tolist()),
        season_margin_games=len(new_margins),week5={
          model:probability_diagnostics(
            legs.loc[legs.model.eq(model)&legs.week.eq(5)],
            tickets.loc[tickets.model.eq(model)&tickets.week.eq(5)])
          for model in VARIANTS
        },totals={model:probability_diagnostics(
            legs.loc[legs.model.eq(model)],tickets.loc[tickets.model.eq(model)])
            for model in VARIANTS},
        warning=[
          "Historic odds are NOT timestamped executable prices.",
          "The best ML and best ATS side of each game are combined; all four underlying market sides are separately audited.",
          "No same-game legs, money stakes, bankroll limits, or compulsory wagers.",
          "ATS forecasts use empirical score-margin residuals from 2019-2025 out-of-sample games, not 2026 game results.",
          "ATS pushes void their own leg; for market screenshots an actual book may have different parlays/house rules.",
          "2026 W6 includes only completed Thursday game; Sunday remains unforecast in this historical replay.",
        ])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    legs.to_csv(out/"all_four_game_market_sides.csv",index=False)
    tickets.to_csv(out/"mixed_parlay_ticket_audit.csv",index=False)
    weeks.to_csv(out/"2026_weekly_market_comparison.csv",index=False)
    # Leaders include UNQUALIFIED combinations, preventing false omission.
    top=tickets.sort_values(["season","week","model","conservative_ev"],
                            ascending=[True,True,True,False]).groupby(
                            ["season","week","model"]).head(20)
    top.to_csv(out/"top_twenty_even_if_rejected.csv",index=False)
    (out/"report.json").write_text(json.dumps(summary,indent=2))
    print("2026 CROSS-MARKET PARLAY FUNNEL\n"+weeks.to_string(index=False),flush=True)
    print("WEEK FIVE NEAR-QUALIFIERS\n"+
          top.loc[top.week.eq(5)&top.model.eq("market_home_prob")][
              ["selections","conservative_all_win_probability","conservative_ev","status"]
          ].head(10).to_string(index=False),flush=True)
    print("SUMMARY\n"+json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
