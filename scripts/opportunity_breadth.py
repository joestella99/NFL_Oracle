"""Opportunity-breadth allocation experiment.

Keep the $100 weekly target and no fixed total bet-count ceiling. When only
one fragile market passes the formal edge gate, diversify into additional
best-priced, distinct games instead of placing the entire $100 on that one
uncertain pick. Uses pregame features and ranking exclusively.

This is an explicitly separate experiment; does NOT silently replace the
historical production selector.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.roi_full_budget_parlays import distribute_cents

MIN_DISTINCT_GAMES=3
MAX_SINGLE_GAME_SHARE=.40
BUDGET=100.


def capped_weights(weights, total_cents, cap_cents):
    raw=np.asarray(weights,float)
    if len(raw)*cap_cents<total_cents:
        raise ValueError("Too few distinct games to enforce allocation cap")
    raw=np.maximum(0,raw)
    if raw.sum()==0:raw=np.ones(len(raw))
    allocation=np.zeros(len(raw),dtype=int)
    remaining=total_cents
    active=np.ones(len(raw),bool)
    # Iterative water filling in whole cents; allocation cap is per GAME,
    # not a ban on adding more qualifying games.
    while remaining>0:
        ix=np.where(active)[0]
        if not len(ix):raise AssertionError("Ran out of eligible games")
        w=raw[ix]
        if w.sum()<=0:w=np.ones(len(ix))
        suggested=distribute_cents(w,remaining/100.)
        proposed=(suggested*100).astype(int)
        capacity=cap_cents-allocation[ix]
        blocked=proposed>=capacity
        if blocked.any():
            winners=ix[blocked]
            allocation[winners]+=capacity[blocked]
            remaining-=int(capacity[blocked].sum())
            active[winners]=False
        else:
            allocation[ix]+=proposed
            remaining=0
    return allocation/100.


def allocate_week(market,budget=BUDGET):
    records=[]
    for key,slate in market.groupby(["policy","season","week"]):
        best=(slate.sort_values(["risk_score","estimated_ev","game_id","kind","side"],
                       ascending=[False,False,True,True,True],kind="stable")
              .drop_duplicates("game_id"))
        if best.empty:continue
        qualified=best[best.qualifies].copy()
        n=max(MIN_DISTINCT_GAMES,len(qualified))
        # If market strength is scarce, promote highest-ranked rejected
        # opportunities but never mislabel them as positive expected value.
        combined=pd.concat([qualified,best[~best.game_id.isin(qualified.game_id)]],
                           ignore_index=True).head(n).copy()
        total=int(round(budget*100))
        capacity=int(round(MAX_SINGLE_GAME_SHARE*total))
        if len(combined)*capacity < total:
            # NFL byes can create very small slates; in that case required
            # concentration is a disclosed unavoidable exception.
            capacity=int(np.ceil(total/len(combined)))
        weights=(np.maximum(.05,combined.win_prob.to_numpy(float))**2
                 *np.exp(2*np.clip(combined.risk_score.to_numpy(float),-.2,.2)))
        combined["stake"]=capped_weights(weights,total,capacity)
        combined["selected_below_gate"]=~combined.qualifies
        combined["profit"]=np.where(combined.push,0.,
                 np.where(combined.won,combined.stake*combined.payout,-combined.stake))
        combined["allocator"]="opportunity_breadth"
        if abs(combined.stake.sum()-budget)>.005:raise AssertionError("Budget mismatch")
        records.append(combined)
    return pd.concat(records,ignore_index=True)


def compare(new,legacy):
    rows=[]
    for policy in sorted(new.policy.unique()):
        for label,start,end in [("2016-2025",2016,2025),("2016-2022",2016,2022),
                                 ("2023-2025",2023,2025),("2025",2025,2025)]:
            n=new.loc[new.policy.eq(policy)&new.season.between(start,end)]
            o=legacy.loc[legacy.policy.eq(policy)&legacy.season.between(start,end)]
            rows.append(dict(policy=policy,period=label,
                bets=len(n),weeks=n[["season","week"]].drop_duplicates().shape[0],
                below_gate_count=int(n.selected_below_gate.sum()),
                max_game_stake=float(n.stake.max()),
                new_staked=round(float(n.stake.sum()),2),
                new_profit=round(float(n.profit.sum()),2),
                new_roi=round(float(n.profit.sum()/n.stake.sum()),5),
                baseline_profit=round(float(o.profit.sum()),2),
                baseline_roi=round(float(o.profit.sum()/o.stake.sum()),5)))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--market-checks",required=True)
    p.add_argument("--baseline",required=True)
    p.add_argument("--output",default="results/opportunity_breadth")
    a=p.parse_args()
    market=pd.read_csv(a.market_checks)
    baseline=pd.read_csv(a.baseline)
    chosen=allocate_week(market)
    summary=compare(chosen,baseline)
    dst=Path(a.output);dst.mkdir(parents=True,exist_ok=True)
    chosen.to_csv(dst/"bets.csv",index=False)
    summary.to_csv(dst/"comparison.csv",index=False)
    (dst/"report.json").write_text(json.dumps({
        "comparison":summary.to_dict("records"),
        "description":"Three distinct games minimum when possible; 40% per-game normal maximum; fills from weaker EV games when needed to meet forced $100 budget.",
        "limitations":["Historical odds lack executable timestamps.",
          "Concentration can be reduced without creating a market edge.",
          "Forced inclusion of negative-EV opportunities can make performance worse.",
          "This is retrospective and has not been promoted to production."]
    },indent=2))
    print(summary.to_string(index=False))
if __name__=="__main__":main()
