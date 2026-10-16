"""Strictly-as-of nflreadpy injury features for historical NFL parlay evaluation.

QB depth charts and ESPN *current* pages are not assumed to be historical
starting-QB confirmation. nflverse injury.date_modified records may be the
latest modification rather than first publication. Filter at kickoff, flag
missing coverage, and retain a separate source/coverage audit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.export_injury_reports import extract

FIELDS = (
    "qb_out_diff", "ol_out_diff", "qb_questionable_diff", "ol_questionable_diff",
    "qb_report_coverage", "ol_report_coverage",
    "qb_depth_projection_change_diff", "qb_depth_projection_coverage",
)
# "depth projection" remains unavailable unless an as-of snapshot is provided.
SOURCE_KIND = "nflreadpy_injuries_asof"

def from_nflreadpy(games: pd.DataFrame, injuries: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Return one game-level row, including unknown / missingness indicators.

    Uses the canonical NFL schedules for home/away team mapping, not string
    parsing game IDs. Missing reports are NOT 'healthy'; coverage=0.
    """
    required = {"game_id","home_team","away_team","season","week",
                "gameday","gametime","game_type"}
    if not required.issubset(games):
        raise ValueError(f"Schedule missing {sorted(required-set(games))}")
    games = games.loc[games.game_type.eq("REG")].copy()
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate games")
    if injuries.empty:
        raw = pd.DataFrame(columns=["game_id","team","role","status_known",
            "out_count","questionable_count","doubtful_count"])
        audit = pd.DataFrame(columns=["game_id","team","potential_reports",
            "reports_pre_kickoff","reports_excluded_as_late"])
    else:
        raw,audit = extract(games,injuries)
    lookup = {
        (str(r.game_id),str(r.team),str(r.role)):
        r for r in raw.itertuples(index=False)
    }
    results=[]
    for game in games.itertuples(index=False):
        item={"game_id":str(game.game_id)}
        for role in ("qb","ol"):
            a=lookup.get((str(game.game_id),str(game.away_team),role))
            h=lookup.get((str(game.game_id),str(game.home_team),role))
            covered=(a is not None and h is not None and
                    bool(a.status_known) and bool(h.status_known))
            item[f"{role}_report_coverage"]=int(covered)
            for field,suffix in (("out_count","out"),("questionable_count","questionable")):
                item[f"{role}_{suffix}_diff"]=(
                    float(getattr(a,field)-getattr(h,field)) if covered else 0.)
        item["qb_depth_projection_change_diff"]=0.
        item["qb_depth_projection_coverage"]=0
        results.append(item)
    return pd.DataFrame(results,columns=["game_id",*FIELDS]),audit


def source_summary(features: pd.DataFrame,audit: pd.DataFrame) -> dict:
    return {
        "games":int(len(features)),
        "both_team_qb_report_games":int(features.qb_report_coverage.sum()) if len(features) else 0,
        "both_team_ol_report_games":int(features.ol_report_coverage.sum()) if len(features) else 0,
        "qb_depth_confirmed_games":0,
        "late_modified_player_report_rows":int(audit.reports_excluded_as_late.sum()) if len(audit) else 0,
        "source":SOURCE_KIND,
        "warning":"Only last-modified pre-kickoff injury reports enter the model. 'QB' reports do not establish starter identity. Zero change with zero coverage means unknown, not healthy. nflreadpy depth-chart weeks are not certified first-publication timestamps.",
    }
