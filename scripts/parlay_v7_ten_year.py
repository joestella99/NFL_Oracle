"""V7 ten-season as-of replay over the *same* V5 frozen market-side audits.

Selection uses past observations only. Published returns are proxy prices,
not accepted sportsbook tickets. Historical development seasons not untouched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nfl_oracle.v7 import V7Policy, calibrate_legs, tickets_for_slate


def run(audits: pd.DataFrame, policy=V7Policy()):
    required={"season","week","model","game_id","kind","side",
              "win_probability","odds_decimal","won","pushed","push_probability"}
    missing=required-set(audits.columns)-({"odds_decimal"} if "decimal_odds" in audits.columns else set())
    if missing:
        raise ValueError(f"Missing audit columns {sorted(missing)}")
    data=audits.copy()
    if "odds_decimal" not in data and "decimal_odds" in data:
        data=data.rename(columns={"decimal_odds":"odds_decimal"})
    data["won"]=data["won"].replace({"True":True,"False":False,"true":True,"false":False})
    data["pushed"]=data["pushed"].replace({"True":True,"False":False,"true":True,"false":False})
    if data.duplicated(["season","week","model","game_id","kind","side"]).any():
        raise ValueError("Duplicate model-side observations")
    data["bookmaker"]="HISTORICAL_PROXY_NOT_SAME_BOOK"
    audited=[]; tickets=[]; summaries=[]
    for (season,week),slate in data.groupby(["season","week"],sort=True):
        prior=data.loc[(data.season < season) | ((data.season == season) & (data.week < week))]
        # Pass only earlier rows; calibration also enforces chronological cutoffs.
        assessed=calibrate_legs(slate,prior,policy,int(season),int(week))
        audited.append(assessed)
        combos=tickets_for_slate(assessed,policy)
        if combos.empty:
            continue
        results={(str(x.game_id),str(x.side),str(x.kind),str(x.model)):
                 (x.won,x.pushed,float(x.odds_decimal))
                 for x in slate.itertuples(index=False)}
        for idx,t in combos.iterrows():
            outcome=1.
            complete=True
            for item in t.selections.split(" | "):
                game,side,kind=item.rsplit(":",2)
                won,pushed,decimal=results[(game,side,kind,t.model)]
                if pd.isna(won) and pd.isna(pushed):
                    complete=False
                    break
                outcome*=1. if bool(pushed) else decimal if bool(won) else 0.
            combos.at[idx,"true_unit_return"]=outcome-1 if complete else float("nan")
        tickets.append(combos)
    legs=pd.concat(audited,ignore_index=True) if audited else pd.DataFrame()
    ts=pd.concat(tickets,ignore_index=True) if tickets else pd.DataFrame()
    if len(ts):
        for model,g in ts.groupby("model"):
            qualified=g.loc[g.status.eq("RESEARCH_QUALIFIED")]
            leader=(qualified.sort_values(["conservative_ev","conservative_all_win_probability"],
                        ascending=False).drop_duplicates(["season","week"]))
            for name,sub in [("all_qualifying",qualified),("one_ticket_per_week",leader)]:
                summaries.append(dict(model=model,view=name,
                    tickets=len(sub),weeks=sub[["season","week"]].drop_duplicates().shape[0],
                    profitable_rate=float(sub.true_unit_return.gt(0).mean()) if len(sub) else None,
                    mean_predicted_all_win=float(sub.conservative_all_win_probability.mean()) if len(sub) else None,
                    equal_unit_roi=float(sub.true_unit_return.mean()) if len(sub) else None))
    return legs,ts,pd.DataFrame(summaries)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit",required=True)
    p.add_argument("--output",default="results/parlay_v7_ten_year")
    a=p.parse_args()
    legs,ts,summary=run(pd.read_csv(a.audit))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    legs.to_csv(out/"market_side_decisions.csv",index=False)
    ts.to_csv(out/"ticket_audit.csv",index=False)
    summary.to_csv(out/"comparison.csv",index=False)
    report=dict(status="TEN_YEAR_HISTORICAL_PROXY_RESEARCH",legs=len(legs),
        evaluated_tickets=len(ts),qualifying_tickets=int(ts.status.eq("RESEARCH_QUALIFIED").sum()) if len(ts) else 0,
        summary=summary.to_dict("records"),
        limitations=["Historical prices are not verified executable same-book tickets.",
          "2016-2025 were previously used during model development.",
          "Different games treated as independent; overlapping tickets are not independent observations.",
          "Calibration stress is a heuristic, not statistical lower confidence bound.",
          "This replays frozen V5 pregame probabilities, NOT a new V7 football forecaster."])
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
