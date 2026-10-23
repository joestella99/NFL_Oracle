"""V7 research CLI over timestamped live/historical leg audits.

No model fitting on outcomes of the scored slate. Use only recorded pregame
observations for historical calibration; source pricing remains an explicit risk.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from nfl_oracle.v7 import V7Policy, calibrate_legs, tickets_for_slate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True, help="Pregame market_side_audit.csv")
    ap.add_argument("--history", required=True, help="Earlier season/weeks market-side audit WITH known outcomes")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--output", default="results/v7_decisions")
    args=ap.parse_args()
    policy=V7Policy()
    live=pd.read_csv(args.legs)
    history=pd.read_csv(args.history)
    if len(live) and (
        not live.season.eq(args.season).all() or
        not live.week.eq(args.week).all()
    ):
        raise ValueError("Live audit contains different games than requested season/week")
    audited=calibrate_legs(live,history,policy,args.season,args.week)
    tickets=tickets_for_slate(audited,policy)
    path=Path(args.output)
    path.mkdir(parents=True,exist_ok=True)
    audited.to_csv(path/"legs_audit.csv",index=False)
    tickets.to_csv(path/"tickets_audit.csv",index=False)
    report=dict(version="v7-research-guarded",season=args.season,week=args.week,
                policy=asdict(policy),market_sides=len(audited),
                candidate_sides=int(audited.v7_leg_status.eq("CANDIDATE").sum())
                if len(audited) else 0,
                qualifying_tickets=int(tickets.status.eq("RESEARCH_QUALIFIED").sum())
                if len(tickets) else 0,
                limitations=["Not the full historical football/medical pipeline.",
                             "Historical market snapshots may not be executable prices.",
                             "Probability stress is heuristic, not a confidence interval.",
                             "Parlay legs assume independent games.",
                             "Model advantage has not been demonstrated prospectively."])
    (path/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
