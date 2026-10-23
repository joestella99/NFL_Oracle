import pandas as pd
import pytest
from scripts.v7_walkforward import replay
from nfl_oracle.v7 import V7Policy


def create_audit(n=240):
    rows=[]
    for i in range(n):
        rows.append(dict(season=2025,week=1,game_id=f"g{i}",model="football",
          kind="ML",side="home",win_probability=.75,won=i%4!=0,
          odds_decimal=1.6,push_probability=0.,pushed=False))
    rows.append(dict(season=2026,week=2,game_id="upcoming",model="football",
      kind="ML",side="away",win_probability=.75,won=False,
      odds_decimal=1.6,push_probability=0.,pushed=False))
    return pd.DataFrame(rows)


def test_walk_forward_sees_only_prior_settled_games():
    decisions,summary=replay(create_audit())
    later=decisions.loc[decisions.game_id.eq("upcoming")].iloc[0]
    assert later.calibration_samples==240
    assert later.v7_leg_status=="CANDIDATE"
    assert later.realized_unit_return==-1
    assert summary.iloc[0].candidate_legs==1


def test_duplicate_game_side_is_blocked():
    data=create_audit()
    with pytest.raises(ValueError):
        replay(pd.concat([data,data.tail(1)],ignore_index=True))


def test_early_history_abstains():
    choices,summary=replay(create_audit(20))
    assert choices.iloc[-1].v7_leg_status=="ABSTAIN_NO_CALIBRATION"
    assert summary.iloc[0].candidate_legs==0
