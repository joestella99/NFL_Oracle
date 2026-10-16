"""Frozen checkpoint prevents accidental retrospective strategy replacement."""
from pathlib import Path
import pandas as pd
import pytest
from scripts.checkpoint_v1 import freeze

def test_freeze_checks_full_weekly_budget(tmp_path):
    inp=tmp_path/"in";inp.mkdir()
    rows=[]
    for year in range(2016,2026):
        for week in range(1,19 if year>=2021 else 18):
            rows.append(dict(policy="risk_tolerant",season=year,week=week,
                             game_id=f"{year}_{week}",stake=100.,profit=20.,
                             kind="ML",side="home"))
    # This fixture may intentionally be 175 rather than 180 weeks; enforce
    # exact total as well as budget reconciliation.
    pd.DataFrame(rows).to_csv(inp/"straight_bets.csv",index=False)
    pd.DataFrame([dict(policy="risk_tolerant",season=2021,week=1,
        game_a="x",game_b="y",stake=5.,profit=-5.)]).to_csv(
        inp/"calculated_cross_game_parlays.csv",index=False)
    for f in ("annual_comparison.csv","weekly_comparison.csv","ten_year_totals.csv"):
        pd.DataFrame([dict(policy="risk_tolerant",combined_staked=1,
                           combined_net=0)]).to_csv(inp/f,index=False)
    with pytest.raises(AssertionError):
        freeze(inp,tmp_path/"out")

def test_freeze_fails_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        freeze(tmp_path,tmp_path/"out")
