import pandas as pd
import pytest
from nfl_oracle.v7 import V7Policy, calibrate_legs, tickets_for_slate


def history(n=240):
    return pd.DataFrame([dict(season=2025,week=1,model="football",
        kind="ML",win_probability=.75,won=i%4 != 0) for i in range(n)])


def leg(game,side="home",p=.75,odds=1.6):
    return dict(game_id=game,season=2026,week=6,model="football",
        bookmaker="book1",kind="ML",side=side,
        win_probability=p,odds_decimal=odds,push_probability=0.)


def test_future_results_cannot_calibrate():
    h=history()
    h=pd.concat([h,pd.DataFrame([dict(season=2026,week=6,model="football",
        kind="ML",win_probability=.75,won=False)]*999)],ignore_index=True)
    out=calibrate_legs(pd.DataFrame([leg("a")]),h,asof_season=2026,asof_week=6)
    assert out.iloc[0].calibration_samples==240
    assert out.iloc[0].bucket_samples==240
    assert out.iloc[0].estimated_calibrated_probability==pytest.approx(.75)


def test_missing_calibration_abstains():
    o=calibrate_legs(pd.DataFrame([leg("a")]),history(10),asof_season=2026,asof_week=6)
    assert o.iloc[0].v7_leg_status=="ABSTAIN_NO_CALIBRATION"
    assert tickets_for_slate(o).empty


def test_expensive_favorite_rejected_even_if_likely():
    o=calibrate_legs(pd.DataFrame([leg("a",odds=1.1)]),history(),asof_season=2026,asof_week=6)
    assert o.iloc[0].v7_leg_status=="NO_VERIFIED_EDGE"


def test_both_sides_and_distinct_games():
    h=history()
    legs=pd.DataFrame([leg("a","home"),leg("a","away"),leg("b","home")])
    evaluated=calibrate_legs(legs,h,asof_season=2026,asof_week=6)
    ts=tickets_for_slate(evaluated)
    assert len(ts)==2
    assert all("a:home" in s or "a:away" in s for s in ts.selections)
    assert all("b:home" in s for s in ts.selections)
    assert all("a:home:ML | a:away:ML" not in s for s in ts.selections)


def test_cannot_score_invalid_probability():
    o=calibrate_legs(pd.DataFrame([leg("a",p=1.5)]),history(),asof_season=2026,asof_week=6)
    assert o.iloc[0].v7_leg_status=="INVALID_PROBABILITY"


def test_single_game_candidate_has_no_parlay():
    evaluated=calibrate_legs(pd.DataFrame([leg("only-game")]),history(),asof_season=2026,asof_week=6)
    assert evaluated.iloc[0].v7_leg_status=="CANDIDATE"
    assert tickets_for_slate(evaluated).empty
