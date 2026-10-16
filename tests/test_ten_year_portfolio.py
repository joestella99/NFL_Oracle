"""Ten-season reporting integrity and historical warmup requirements."""
import numpy as np
import pandas as pd
import pytest
from scripts.simulate_ten_year_portfolio import weekly_comparison, _summary, run

def _single(game,week,stake,profit,policy="risk_tolerant"):
    return dict(game_id=game,season=2016,week=week,policy=policy,kind="ML",
      stake=stake,profit=profit,forced_fallback=False)
def test_weekly_straights_and_parlays_are_side_by_side():
    singles=pd.DataFrame([_single("a",1,60.,12.),_single("b",1,40.,-40.),
         _single("c",2,100.,20.)])
    parlays=pd.DataFrame([dict(policy="risk_tolerant",season=2016,week=1,
         game_a="d",game_b="e",stake=5.,profit=7.5,both_won=True)])
    weekly=weekly_comparison(singles,parlays)
    assert len(weekly)==2
    a=weekly[weekly.week.eq(1)].iloc[0]
    assert a.straight_staked==100
    assert a.straight_net==-28
    assert a.parlay_staked==5
    assert a.parlay_net==7.5
    assert a.combined_staked==105
    assert a.combined_net==-20.5
    assert np.isclose(a.combined_roi_pct,100*(-20.5/105))
    b=weekly[weekly.week.eq(2)].iloc[0]
    assert b.parlay_tickets==0 and b.parlay_staked==0
    assert b.combined_staked==100 and b.combined_net==20

def test_ten_year_replay_refuses_missing_pregame_warmup():
    with pytest.raises(ValueError,match="warmup"):
        run(pd.DataFrame({"game_id":["a"],"season":[2016],"model":["baseline"],
                          "prediction":[.8]}),pd.DataFrame())
