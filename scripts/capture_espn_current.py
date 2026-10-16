"""Capture *present-time* ESPN NFL injury/depth-chart views for future games.

The Public-ESPN-API repo documents unofficial ESPN endpoints, not a reliable
historical pregame snapshot archive. Persist timestamp + URL + source payload.
Never backdate a captured value to a game already kicked off. Snapshots are
NOT matched to historical 2023-2025 games for training.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

ROOT="https://site.api.espn.com/apis/site/v2/sports/football/nfl"
SOURCE_DOCS="https://github.com/pseudo-r/Public-ESPN-API"

def fetch(path: str, *, timeout: int=12) -> dict:
    url=f"{ROOT}/{path}"
    now=datetime.now(timezone.utc).isoformat()
    item=dict(url=url,observed_at=now,source_docs=SOURCE_DOCS)
    try:
        req=Request(url,headers={
            "User-Agent":"NFL-Oracle-research/0.1 (public NFL information)",
            "Accept":"application/json"})
        with urlopen(req,timeout=timeout) as response:
            payload=json.load(response)
        item.update(ok=True,payload=payload)
    except (TimeoutError,HTTPError,URLError,OSError,ValueError) as exc:
        item.update(ok=False,error_type=type(exc).__name__,message=str(exc)[:200])
    return item


def capture() -> dict:
    team_list=fetch("teams")
    injuries=fetch("injuries")
    records=[team_list,injuries]
    teams=[]
    if team_list["ok"]:
        source=team_list["payload"]
        for league in source.get("sports",[]):
            for conf in league.get("leagues",[]):
                for row in conf.get("teams",[]):
                    team=row.get("team",{})
                    if str(team.get("id","")).isdigit():
                        teams.append(str(team["id"]))
    for team_id in sorted(set(teams),key=int):
        records.append(fetch(f"teams/{team_id}/depthcharts"))
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
        documentation=SOURCE_DOCS,source="unofficial_current_ESPN",
        endpoint_count=len(records),
        succeeded=sum(x["ok"] for x in records),
        all_snapshots=records,
        limitations=[
            "Observed timestamp is WHEN NFL Oracle fetched a snapshot, NOT the original ESPN publication timestamp.",
            "An ESPN depth chart projects a starter; it does not conclusively identify the actual starting QB.",
            "Only games kicking off AFTER observed_at may use this data prospectively.",
            "No historical replay backfill is allowed using these current endpoints.",
            "Endpoint definitions are undocumented/unstable and may be blocked or empty.",
        ])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/espn_asof_2026")
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    data=capture()
    (out/"snapshot.json").write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!="all_snapshots"},indent=2))
    if data["succeeded"]==0:
        print("WARNING: no ESPN snapshot acquired. No historical signal inferred.")

if __name__=="__main__":
    main()
