"""Exhaustive same-week round-robin moneyline experiment.

Every subset of length 2..N is evaluated: NO fixed maximum legs or ticket
count; combinations can overlap with straight bets and each other. Unique
game per ticket, no same-game parlay quote invented.

All historical leg prices are compounded mathematically, never represented
as proven simultaneous single-book quotes. All stakes share a user-configurable
*bankroll* budget, not separate per-ticket maxima.

Only +EV tickets should be considered as value. A separate SMALL exploratory
sleeve optionally tests nonpositive-EV tickets; it is not a recommendation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.roi_full_budget_parlays import distribute_cents, compounded_parlay_decimal

# Not leg or bet-count limits. These are evidence requirements.
MIN_SINGLE_WIN_PROB = .60
MIN_JOINT_WIN_PROB = .10
PROBABILITY_HAIRCUT = .01
BASE_RESEARCH_BANKROLL = 1000.
EXPLORATION_WEEKLY_BANKROLL_FRACTION = .005    # 0.5%, not $15 flat
POSITIVE_VALUE_FRACTIONAL_KELLY = .25          # fractional Kelly, clipped by bankroll
MAX_PARLAY_BANKROLL_FRACTION = 1.              # only absolute bankroll is the bound


def generate_round_robins(slate, *, min_leg_prob=MIN_SINGLE_WIN_PROB,
                          min_joint_prob=0.0):
    """No limit on leg counts. Enumerate every plausible subset.

    Finite NFL slates permit exhaustive examination. A zero default gate
    retains long-shot combinations for EV analysis, not automatic funding.
    """
    ml = (
        slate.loc[slate.kind.eq("ML")]
        .sort_values(["win_prob","game_id","side"],ascending=[False,True,True],kind="stable")
        .drop_duplicates("game_id")
    )
    ml = ml[ml.win_prob.ge(min_leg_prob)]
    legs = ml.to_dict("records")
    considered = 0
    qualifying = []
    for size in range(2, len(legs)+1):
        for combo in combinations(legs,size):
            considered += 1
            p = float(math.prod(x["win_prob"] for x in combo))
            if p < min_joint_prob:
                continue
            # Conservative per-leg downshift, not retrospective per-ticket fitting.
            q = float(math.prod(max(0., x["win_prob"] - PROBABILITY_HAIRCUT)
                                for x in combo))
            decimal = compounded_parlay_decimal(*(x["odds"] for x in combo))
            ev = p * decimal - 1.
            conservative_ev = q * decimal - 1.
            qualifying.append({
                "season": combo[0]["season"],
                "week": combo[0]["week"],
                "policy": combo[0]["policy"],
                "leg_count":size,
                "games":" | ".join(str(x["game_id"]) for x in combo),
                "sides":" | ".join(str(x["side"]) for x in combo),
                "american_odds_legs":" | ".join(str(int(x["odds"])) for x in combo),
                "leg_probabilities":" | ".join(f'{x["win_prob"]:.4f}' for x in combo),
                "joint_probability":p,
                "conservative_joint_probability":q,
                "calculated_decimal_odds":decimal,
                "estimated_ev":ev,
                "conservative_ev":conservative_ev,
                "all_won":bool(all(x["won"] for x in combo)),
                "leg_game_ids":tuple(str(x["game_id"]) for x in combo),
                "research_only":True,
            })
    return qualifying, considered, len(legs)


def _size_sleeve(tickets, bankroll, mode):
    """Adaptive evidence-linked parlay sleeve.

    Positive evidence scales with fractional Kelly. A small optional
    exploratory budget is reserved for studying candidate collections;
    it never pretends negative EV represents value.
    """
    if bankroll<=0 or not len(tickets):
        return 0.
    if mode=="positive_only":
        qualifying=[x for x in tickets if x["conservative_ev"]>0]
        if not qualifying:return 0.
        # Kelly fraction and payout must refer to THE SAME candidate.
        # Taking the highest edge and highest odds independently is invalid.
        best=max(qualifying,key=lambda x: x["conservative_ev"] /
                 max(.01,x["calculated_decimal_odds"]-1))
        fractional_edge=best["conservative_ev"] / max(.01,best["calculated_decimal_odds"]-1)
        return round(min(bankroll,bankroll * POSITIVE_VALUE_FRACTIONAL_KELLY *
                         fractional_edge),2)
    if mode=="exploratory":
        # 0.5% of a hypothetical $1k parlay bankroll, with NO requirement
        # to spend that amount if no combos meet the research threshold.
        return round(bankroll*EXPLORATION_WEEKLY_BANKROLL_FRACTION,2)
    raise ValueError(mode)


def allocate_round_robins(tickets, budget, mode):
    """No max tickets; penny budget mathematically limits actual funded tickets."""
    if budget<=0 or not tickets:
        return []
    if mode=="positive_only":
        eligible=[x for x in tickets if x["conservative_ev"]>0]
    elif mode=="exploratory":
        eligible=[x for x in tickets if x["estimated_ev"]>=-.10
                  and x["joint_probability"]>=MIN_JOINT_WIN_PROB]
    else:raise ValueError(mode)
    if not eligible:return []
    # Portfolio-wide cents: overlapping combinations share ONE exposure budget.
    # Highest evidence prioritized; no fixed count cutoff.
    eligible=sorted(eligible,key=lambda x:(-x["estimated_ev"],-x["joint_probability"],x["games"]))
    weights=np.array([
        max(.01, x["joint_probability"]) *
        math.exp(8*min(.25,x["estimated_ev"])) /
        math.sqrt(max(1.,x["calculated_decimal_odds"]))
        for x in eligible],float)
    stakes=distribute_cents(weights,budget)
    selected=[]
    for x,stake in zip(eligible,stakes):
        if stake < .01:continue
        entry={k:v for k,v in x.items() if k!="leg_game_ids"}
        entry.update({"mode":mode,"stake":float(stake),
                      "profit":float(stake*(x["calculated_decimal_odds"]-1)
                                     if x["all_won"] else -stake)})
        selected.append(entry)
    return selected


def simulate(universe,singles,bankroll=BASE_RESEARCH_BANKROLL):
    """Same candidates across modes, avoiding data leakage from actual outcomes."""
    straight_lookup={
        key:set(g.game_id.astype(str))
        for key,g in singles.groupby(["policy","season","week"])
    }
    all_tickets=[]
    weekly=[]
    for key,slate in universe.groupby(["policy","season","week"]):
        candidates, considered, nlegs=generate_round_robins(slate)
        # Count all possible lengths and price levels; none chosen by results.
        favorable=[x for x in candidates if x["conservative_ev"]>0]
        for mode in ("positive_only","exploratory"):
            budget=_size_sleeve(candidates,bankroll,mode)
            picked=allocate_round_robins(candidates,budget,mode)
            funded_games={x["games"] for x in picked}
            direct=straight_lookup.get(key,set())
            for x in picked:
                legs=set(x["games"].split(" | "))
                x["overlaps_straight"]=bool(legs & direct)
                x["week_parlay_budget"]=budget
                all_tickets.append(x)
            weekly.append({
                "policy":key[0],"season":key[1],"week":key[2],"mode":mode,
                "eligible_ml_legs":nlegs,"combinations_considered":considered,
                "viable_probability_combinations":len(candidates),
                "conservative_positive_ev_count":len(favorable),
                "max_legs_considered":nlegs if nlegs>=2 else 0,
                "max_legs_passing_probability_gate":max((x["leg_count"] for x in candidates
                    if x["joint_probability"]>=MIN_JOINT_WIN_PROB),default=0),
                "longest_winning_unfunded_legs":max((x["leg_count"] for x in candidates
                    if x["all_won"] and x["games"] not in funded_games),default=0),
                "tickets_funded":len(picked),"stake":float(sum(x["stake"] for x in picked)),
                "profit":float(sum(x["profit"] for x in picked)),
                "budget":budget,
                "overlap_with_straights_count":sum(x["overlaps_straight"] for x in picked),
            })
    tickets=pd.DataFrame(all_tickets)
    weeks=pd.DataFrame(weekly)
    if len(tickets):
        for _,g in tickets.groupby(["policy","season","week","mode"]):
            allocated=g.stake.sum()
            if allocated>g.week_parlay_budget.iloc[0]+.001:
                raise AssertionError("Parlay wagers exceeded shared weekly risk budget")
        assert not (tickets.games.str.split(" | ",regex=False).map(lambda v:len(v)!=len(set(v)))).any()
    if weeks.empty:
        raise ValueError("No historical weeks found")
    return tickets,weeks


def audit_correlated_exposure(tickets, singles):
    """Quantify correlated round-robin risk from repeating a game across tickets.

    Exposed parlay stake is the sum of all tickets that require the selected
    team to win. If that team loses, that whole exposed amount is lost,
    irrespective of whether other legs were correct. These exposures overlap;
    NEVER sum them to estimate total portfolio risk.
    """
    fields=["policy","season","week","mode","game_id","side",
            "parlay_tickets_containing_leg","parlay_stake_exposed",
            "weekly_parlay_staked","pct_parlay_budget_at_risk",
            "straight_stake_on_same_game","straight_side",
            "parlay_pnl_if_leg_loses"]
    if tickets.empty:
        return pd.DataFrame(columns=fields)
    direct=singles.groupby(["policy","season","week","game_id"]).first()
    rows=[]
    for key,group in tickets.groupby(["policy","season","week","mode"]):
        shared={}
        total=float(group.stake.sum())
        for ticket in group.itertuples(index=False):
            games=ticket.games.split(" | ")
            sides=ticket.sides.split(" | ")
            for game,side in zip(games,sides):
                record=shared.setdefault((game,side),[0,0.])
                record[0]+=1
                record[1]+=float(ticket.stake)
        for (game,side),(count,exposed) in sorted(shared.items()):
            dkey=key[:3]+(game,)
            stake=float(direct.loc[dkey,"stake"]) if dkey in direct.index else 0.
            straight_side=str(direct.loc[dkey,"side"]) if dkey in direct.index else ""
            rows.append({
                "policy":key[0],"season":key[1],"week":key[2],"mode":key[3],
                "game_id":game,"side":side,
                "parlay_tickets_containing_leg":count,
                "parlay_stake_exposed":round(exposed,2),
                "weekly_parlay_staked":round(total,2),
                "pct_parlay_budget_at_risk":exposed/total if total else 0.,
                "straight_stake_on_same_game":stake,
                "straight_side":straight_side,
                "parlay_pnl_if_leg_loses":round(-exposed,2),
            })
    return pd.DataFrame(rows,columns=fields)


def make_report(universe,singles,tickets,weekly,baseline_parlays):
    results=[]
    for policy in sorted(universe.policy.unique()):
        direct=singles.loc[singles.policy.eq(policy)]
        baseline=baseline_parlays.loc[baseline_parlays.policy.eq(policy)] if len(baseline_parlays) else pd.DataFrame()
        for mode in ("positive_only","exploratory"):
            ws=weekly.loc[weekly.policy.eq(policy)&weekly["mode"].eq(mode)]
            bt=tickets.loc[tickets.policy.eq(policy)&tickets["mode"].eq(mode)] if len(tickets) else pd.DataFrame()
            for period,start,end in [("2016-2025",2016,2025),("2016-2022",2016,2022),("2023-2025",2023,2025)]:
                w=ws.loc[ws.season.between(start,end)]
                s=direct.loc[direct.season.between(start,end)]
                b=baseline.loc[baseline.season.between(start,end)] if len(baseline) else pd.DataFrame()
                p=float(w.profit.sum()); st=float(w.stake.sum())
                ss=float(s.stake.sum()); sp=float(s.profit.sum())
                bs=float(b.stake.sum()) if len(b) else 0
                bp=float(b.profit.sum()) if len(b) else 0
                results.append({
                    "policy":policy,"mode":mode,"period":period,
                    "weeks":len(w),"parlay_tickets":int(w.tickets_funded.sum()),
                    "parlay_staked":round(st,2),"parlay_net":round(p,2),
                    "parlay_roi_pct":round(100*p/st,3) if st else None,
                    "straight_staked":round(ss,2),"straight_net":round(sp,2),
                    "straight_roi_pct":round(100*sp/ss,3),
                    "combined_staked":round(ss+st,2),"combined_net":round(sp+p,2),
                    "combined_roi_pct":round(100*(sp+p)/(ss+st),3),
                    "old_one_ticket_parlay_staked":round(bs,2),
                    "old_one_ticket_parlay_net":round(bp,2),
                    "conservative_positive_ev_opportunities":int(w.conservative_positive_ev_count.sum()),
                    "max_legs_considered":int(w.max_legs_considered.max()),
                })
    return {
        "strategy":"No fixed leg or ticket count limit, with overlapping and round-robin combinations; all share one per-week parlay risk sleeve.",
        "research_bankroll":BASE_RESEARCH_BANKROLL,
        "fractional_kelly":POSITIVE_VALUE_FRACTIONAL_KELLY,
        "exploratory_bankroll_fraction":EXPLORATION_WEEKLY_BANKROLL_FRACTION,
        "warnings":[
            "All sportsbook moneyline prices were historical reference odds and not verified as jointly available at the same book.",
            "There is no guarantee of positive expected value; combining negative-EV legs is not an arbitrage.",
            "Ticket win probabilities assume independence across games.",
            "All combinations may correlate heavily through overlapping legs, amplifying drawdown risk.",
            "The 2016-25 sample has been repeatedly explored, so this is research, not untouched backtesting.",
            "A configurable bankroll ceiling is retained, but there are no arbitrary ticket-count or leg-count caps.",
            "Winning unfunded combinations are retrospective diagnostic cases only, NEVER counted as selected bets or profit.",
            "Positive-value mode examines longshots at any joint probability, but only stakes those with positive conservative estimated EV.",
        ],
        "results":results
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--market-checks",required=True)
    p.add_argument("--straight-bets",required=True)
    p.add_argument("--old-parlays",required=True)
    p.add_argument("--output",default="results/ten_year_round_robins")
    p.add_argument("--parlay-bankroll",type=float,default=BASE_RESEARCH_BANKROLL)
    args=p.parse_args()
    universe=pd.read_csv(args.market_checks)
    singles=pd.read_csv(args.straight_bets)
    old=pd.read_csv(args.old_parlays)
    tickets,weekly=simulate(universe,singles,bankroll=args.parlay_bankroll)
    rep=make_report(universe,singles,tickets,weekly,old)
    exposure=audit_correlated_exposure(tickets,singles)
    if len(exposure):
        rep["max_single_leg_parlay_exposure_pct"]=round(
            float(exposure.pct_parlay_budget_at_risk.max())*100,2)
        rep["max_single_game_parlay_stake_exposure"]=round(
            float(exposure.parlay_stake_exposed.max()),2)
    else:
        rep["max_single_leg_parlay_exposure_pct"]=0.
        rep["max_single_game_parlay_stake_exposure"]=0.

    dst=Path(args.output);dst.mkdir(parents=True,exist_ok=True)
    (dst/"report.json").write_text(json.dumps(rep,indent=2))
    pd.DataFrame(rep["results"]).to_csv(dst/"roi_comparison.csv",index=False)
    weekly.to_csv(dst/"weekly_round_robin.csv",index=False)
    exposure.to_csv(dst/"correlated_leg_exposure.csv",index=False)
    if len(tickets):tickets.to_csv(dst/"round_robin_tickets.csv",index=False)
    print(json.dumps(rep,indent=2))


if __name__=="__main__":
    main()
