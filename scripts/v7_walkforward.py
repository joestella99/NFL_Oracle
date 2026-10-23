"""Walk-forward V7 leg screening evaluation, using only earlier weeks for calibration.

Consumes frozen pregame market-side audits with known final results. This is
not a real-money simulation: quoted prices may be historical proxies.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nfl_oracle.v7 import V7Policy, calibrate_legs


NEEDED = {"season","week","game_id","model","kind","side",
          "win_probability","won","odds_decimal","push_probability"}


def replay(audits: pd.DataFrame, policy: V7Policy=V7Policy()):
    missing = NEEDED - set(audits.columns)
    if missing:
        raise ValueError("Historical audit missing columns: "+", ".join(sorted(missing)))
    data=audits.copy()
    if data.duplicated(["season","week","game_id","model","kind","side"]).any():
        raise ValueError("Duplicate side of game in historical market audit")
    data["won"]=data["won"].replace({"True":True,"False":False,"true":True,"false":False})
    if not data["won"].dropna().isin([True,False]).all():
        raise ValueError("Malformed settled 'won' column")
    rows=[]
    for (season,week),slate in data.groupby(["season","week"],sort=True):
        # The function itself performs the prior-only filter.
        rated=calibrate_legs(slate,data,policy,int(season),int(week))
        rated["asof_season"]=int(season)
        rated["asof_week"]=int(week)
        rows.append(rated)
    result=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if result.empty:
        return result, pd.DataFrame()
    priced=result.loc[result.won.notna()].copy()
    priced["realized_unit_return"]=np.where(
        priced.won.eq(True),
        priced.odds_decimal.astype(float)-1,
        np.where(priced.push_probability.astype(float).ge(0) &
                 priced.get("pushed",pd.Series(False,index=priced.index)).eq(True),
                 0.,-1.))
    summary=[]
    for (model,kind),g in priced.groupby(["model","kind"]):
        passed=g.loc[g.v7_leg_status.eq("CANDIDATE")]
        summary.append(dict(model=model,kind=kind,
             settled_legs=int(len(g)),candidate_legs=int(len(passed)),
             historical_equal_unit_roi=(float(passed.realized_unit_return.mean())
                                        if len(passed) else None),
             candidate_hit_rate=(float(passed.won.eq(True).mean())
                                 if len(passed) else None),
             candidate_predicted_win_rate=(float(passed.estimated_calibrated_probability.mean())
                                           if len(passed) else None),
             abstain_count=int(g.v7_leg_status.eq("ABSTAIN_NO_CALIBRATION").sum()),
             no_edge_count=int(g.v7_leg_status.eq("NO_VERIFIED_EDGE").sum())))
    return result,pd.DataFrame(summary)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit",required=True)
    ap.add_argument("--output",default="results/v7_walkforward")
    args=ap.parse_args()
    audit=pd.read_csv(args.audit)
    rated,summary=replay(audit)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rated.to_csv(root/"historical_side_decisions.csv",index=False)
    summary.to_csv(root/"market_replay_summary.csv",index=False)
    report=dict(status="HISTORICAL_PRICE_PROXY_RESEARCH",
                observations=len(rated),results=summary.to_dict("records"),
                warning="Years previously examined for model development, historical leg quotes not verified executable, correlated exposures, no expected profitable edge.")
    (root/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
