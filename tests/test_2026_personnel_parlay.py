"""Leakage and percentage-only 2026 parlay feature tests."""
import numpy as np
import pandas as pd
import pytest
from scripts.personnel_asof import from_nflreadpy,FIELDS
from scripts.parlay_2026_review import (
    PercentageRule,prepare_personnel_model,choose_forecast,percentage_shortlist,
)
from scripts.parlay_portfolio_v2 import FEATURES,MODELS


def _history():
    rows=[]
    for year in range(2014,2027):
        for idx in range(20):
            gid=f"{year}_{idx}"
            p=.57+.02*((idx%5)-2)
            for model in MODELS:
                rows.append(dict(game_id=gid,season=year,week=1+idx//5,
                    model=model,prediction=p,
                    market_home_prob=p,home_win=int(idx%3!=0),
                    home_moneyline=-180,away_moneyline=150,
                    **{col:0.1*((idx%7)-3) for col in FEATURES}))
    return pd.DataFrame(rows)


def test_2026_outcomes_never_used_in_2026_calibration():
    source=_history()
    base=prepare_personnel_model(source,pd.DataFrame(columns=["game_id",*FIELDS]))
    flipped=source.copy()
    flipped.loc[flipped.season.eq(2026),"home_win"]=1-flipped.loc[
        flipped.season.eq(2026),"home_win"]
    after=prepare_personnel_model(flipped,pd.DataFrame(columns=["game_id",*FIELDS]))
    for col in ["football_base_prob","football_injury_prob"]:
        a=base.loc[base.season.eq(2026),col].to_numpy(float)
        b=after.loc[after.season.eq(2026),col].to_numpy(float)
        np.testing.assert_allclose(a,b)


def test_model_selection_ignores_all_2026_labels():
    source=_history()
    b=prepare_personnel_model(source,pd.DataFrame(columns=["game_id",*FIELDS]))
    winner,report=choose_forecast(b)
    changed=b.copy()
    changed.loc[changed.season.eq(2026),"home_win"]=1-changed.loc[
        changed.season.eq(2026),"home_win"]
    winner2,report2=choose_forecast(changed)
    assert winner==winner2
    pd.testing.assert_frame_equal(report,report2)


def test_missing_injuries_not_healthy_and_no_invented_starters():
    g=pd.DataFrame([dict(game_id="2026_01_NE_BUF",home_team="BUF",
        away_team="NE",season=2026,week=1,game_type="REG",
        gameday="2026-09-10",gametime="20:15")])
    f,audit=from_nflreadpy(g,pd.DataFrame())
    assert len(f)==1
    assert not int(f.qb_report_coverage.iloc[0])
    assert not int(f.ol_report_coverage.iloc[0])
    assert not int(f.qb_depth_projection_coverage.iloc[0])
    assert f.qb_out_diff.iloc[0]==0


def test_percentage_only_parlays_have_no_stake_columns():
    raw=[]
    for gid,p in zip("abcd",[.85,.82,.80,.78]):
        raw.append(dict(game_id=gid,season=2026,week=1,home_win=int(gid!="d"),
            football_base_prob=p,football_injury_prob=p,
            market_home_prob=.69,home_moneyline=-200,away_moneyline=160,
            baseline=p,matchup=p,sustainability=p))
    game=pd.DataFrame(raw)
    audit,weekly=percentage_shortlist(game,PercentageRule(),"football_base_prob")
    assert len(audit)==11
    assert weekly.iloc[0].evaluated_combinations==11
    assert not {"stake","bankroll","wager"} & set(audit.columns)
    assert not {"stake","bankroll","wager"} & set(weekly.columns)
    assert set(audit.legs)=={2,3,4}
