"""Selective weekly parlay portfolio: turn clusters of value into opportunities.

A longer parlay is NOT inherently better. Evaluate existing 2..N round-robin
candidates, select only conservative positive-EV ones, prefer short tickets
unless a longer ticket earns its extra risk, and account for shared legs.
The user-selected research bankroll bounds aggregate stakes, not ticket count.
All choices use pregame prices and probability estimates, never realized wins.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from scripts.parlay_round_robin import generate_round_robins

BANKROLL = 1000.
FRACTION = .25


def select_portfolio(slate, bankroll=BANKROLL):
    choices, examined, legs = generate_round_robins(slate)
    worthy = [c for c in choices if c["conservative_ev"] > .02]
    # Prioritize EV per unit of capital and reliable joint hit rate.
    worthy.sort(key=lambda x: (
        -(x["conservative_ev"] / max(.01,x["calculated_decimal_odds"]-1))
          * x["conservative_joint_probability"]**.5,
        x["leg_count"],x["games"]))
    tickets = []
    exposure = {}
    # Aggregate bankroll fraction. No cap on number of picks or legs.
    # Each additional ticket must offer a positive conservative EV, and shares
    # shrink as the exposure to any repeated game increases.
    for item in worthy:
        games = item["leg_game_ids"]
        overlap = max((exposure.get(g,0.) for g in games),default=0.)
        if overlap >= .10 * bankroll:
            continue
        price = item["calculated_decimal_odds"]-1
        base_fraction = FRACTION * item["conservative_ev"] / price
        allowance = min(.10*bankroll-overlap,bankroll*.15)
        stake = min(allowance,bankroll*base_fraction/(1+item["leg_count"]*.12))
        stake = round(stake,2)
        if stake < 1.:
            continue
        row={k:v for k,v in item.items() if k!="leg_game_ids"}
        row["stake"]=stake
        row["profit"]=stake*(price if item["all_won"] else -1)
        row["mode"]="selective_value"
        tickets.append(row)
        for g in games:exposure[g]=exposure.get(g,0.)+stake
    # The parlay sleeve cannot exceed capital even across distinct teams.
    if sum(t["stake"] for t in tickets)>bankroll+.001:
        raise AssertionError("Portfolio bankroll exceeded")
    return tickets, {"examined":examined,"eligible_legs":legs,
                     "positive_value_candidates":len(worthy),
                     "funded":len(tickets)}


def run(market, straights, old):
    picks=[]; weekly=[]
    for (policy,year,week),slate in market.groupby(["policy","season","week"]):
        selected,review=select_portfolio(slate)
        profit=sum(p["profit"] for p in selected)
        staked=sum(p["stake"] for p in selected)
        picks.extend(selected)
        weekly.append({"policy":policy,"season":year,"week":week,
                       "parlay_tickets":len(selected),"parlay_staked":staked,
                       "parlay_profit":profit,**review})
    results=pd.DataFrame(weekly)
    rows=[]
    for policy in results.policy.unique():
        for name,frame in [("new_selective_value",results[results.policy.eq(policy)])]:
            s=straights[straights.policy.eq(policy)]
            b=old[old.policy.eq(policy)] if len(old) else pd.DataFrame()
            spent=float(frame.parlay_staked.sum())
            p=float(frame.parlay_profit.sum())
            ss=float(s.stake.sum());sp=float(s.profit.sum())
            bs=float(b.stake.sum()) if len(b) else 0.
            bp=float(b.profit.sum()) if len(b) else 0.
            rows.append({"policy":policy,"mode":name,"parlay_tickets":int(frame.parlay_tickets.sum()),
              "parlay_staked":round(spent,2),"parlay_profit":round(p,2),
              "parlay_roi_pct":round(100*p/spent,3) if spent else None,
              "straight_staked":round(ss,2),"straight_profit":round(sp,2),
              "combined_profit":round(sp+p,2),
              "combined_roi_pct":round(100*(sp+p)/(ss+spent),3),
              "original_one_parlay_staked":round(bs,2),
              "original_one_parlay_profit":round(bp,2),
              "original_combined_roi_pct":round(100*(sp+bp)/(ss+bs),3)})
    return pd.DataFrame(picks),results,pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--market-checks",required=True)
    p.add_argument("--straight-bets",required=True)
    p.add_argument("--old-parlays",required=True)
    p.add_argument("--output",default="results/selective_parlays")
    a=p.parse_args()
    chosen,weekly,summary=run(pd.read_csv(a.market_checks),
        pd.read_csv(a.straight_bets),pd.read_csv(a.old_parlays))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    chosen.to_csv(output/"tickets.csv",index=False)
    weekly.to_csv(output/"weekly.csv",index=False)
    summary.to_csv(output/"summary.csv",index=False)
    (output/"report.json").write_text(json.dumps({
      "results":summary.to_dict("records"),"policy":"Conservative positive-EV, selective portfolio; overlapping legs use shared exposure checks",
      "limitations":["Historical odds not verified as simultaneous book quotes",
         "Backtested years already inspected; results are not prospective validation",
         "Model probability independence and calibration not validated for parlays"]},indent=2))
    print(summary.to_string(index=False))
if __name__=="__main__":main()
