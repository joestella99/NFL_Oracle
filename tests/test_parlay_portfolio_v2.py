"""Parlay v2 invariants: no arbitrary ticket cap and exposure-aware allocation."""
from itertools import combinations
import numpy as np
import pandas as pd
import pytest

from scripts.parlay_portfolio_v2 import (
    RiskThreshold, allocate_tickets, american_decimal, candidate_legs,
    enumerate_tickets, feature_matrix, football_probabilities,
    prepare_prediction_features, FEATURES, MODELS,
)

def four_favorites():
    teams=["a","b","c","d"]
    out=[]
    for team,p in zip(teams,[.85,.82,.80,.78]):
        out.append(dict(game_id=team,side="home",football_p=p,
            market_p=.69,conservative_p=p-.035,odds=-200,decimal=1.5,
            model_disagreement=.01,model_ev=p*1.5-1,
            conservative_leg_ev=(p-.035)*1.5*.98-1))
    return out

def test_four_favorites_produce_all_eleven_distinct_parlays():
    legs=four_favorites()
    tickets=enumerate_tickets(legs,RiskThreshold())
    assert len(tickets)==11
    assert tickets.legs.value_counts().to_dict()=={2:6,3:4,4:1}
    assert all(len(set(g))==len(g) for g in tickets.game_ids)
    assert {"ELIGIBLE_RESEARCH"}==set(tickets.status)

def test_allocation_can_fund_multiple_tickets_without_exceeding_risk():
    risk=RiskThreshold(bankroll=1000)
    tickets=allocate_tickets(enumerate_tickets(four_favorites(),risk),risk)
    picked=tickets[tickets.stake>0]
    assert len(picked)>1
    assert picked.stake.sum()<=50+.001
    assert picked.stake.max()<=5+.001
    for game in ["a","b","c","d"]:
        assert picked.loc[picked.game_ids.map(lambda ids:game in ids),"stake"].sum()<=20+.001

def test_preexisting_straight_game_exposure_reduces_parlay_room():
    risk=RiskThreshold(bankroll=1000)
    tickets=allocate_tickets(enumerate_tickets(four_favorites(),risk),risk,
                             {"a":20.})
    assert (tickets.loc[tickets.game_ids.map(lambda ids:"a" in ids),"stake"]==0).all()
    assert tickets.stake.sum()>0

def test_price_risk_and_ev_screen_reject_unfavorable_tickets():
    risk=RiskThreshold(min_conservative_ev=.99)
    tickets=allocate_tickets(enumerate_tickets(four_favorites(),risk),risk)
    assert tickets.stake.sum()==0
    assert not tickets.status.eq("ELIGIBLE_RESEARCH").any()

def test_refuse_to_invent_odds():
    for price in [0,-110.5+100,99,float("nan")]:
        with pytest.raises(ValueError):
            american_decimal(price)
    assert american_decimal(-200)==1.5
    assert american_decimal(200)==3.

def test_leg_selector_can_include_games_with_straight_bets():
    slates=[]
    for x,p in zip("abcd",[.85,.82,.80,.78]):
        slates.append(dict(game_id=x,football_home_prob=p,market_home_prob=.70,
            home_moneyline=-200,away_moneyline=175,
            baseline=p,matchup=p,sustainability=p))
    legs=candidate_legs(pd.DataFrame(slates),RiskThreshold())
    assert len(legs)==4
    assert set(x["game_id"] for x in legs)==set("abcd")

def test_season_training_does_not_see_current_year_outcomes():
    rng=np.random.default_rng(71)
    rows=[]
    for season in [2014,2015,2016]:
        for idx in range(85):
            game=f"{season}-{idx}"
            p=.2+(.6*(idx%17)/16)
            side=int(rng.random()<p)
            for name in MODELS:
                rows.append(dict(game_id=game,season=season,week=1+idx//6,
                    model=name,prediction=p,market_home_prob=p,
                    home_win=side,home_moneyline=-110,away_moneyline=-110,
                    **{f:float(idx%9)/10 for f in FEATURES}))
    raw=pd.DataFrame(rows)
    original=prepare_prediction_features(raw)
    forecasts=football_probabilities(original,first_season=2016)
    assert forecasts.loc[forecasts.season.eq(2016),"football_home_prob"].notna().all()
    assert forecasts.loc[forecasts.season.lt(2016),"football_home_prob"].isna().all()
    edited=raw.copy()
    edited.loc[edited.season.eq(2016),"home_win"]=1-edited.loc[
        edited.season.eq(2016),"home_win"]
    changed=football_probabilities(prepare_prediction_features(edited),first_season=2016)
    np.testing.assert_allclose(
        forecasts.loc[forecasts.season.eq(2016),"football_home_prob"],
        changed.loc[changed.season.eq(2016),"football_home_prob"])

def test_duplicate_games_in_ticket_rejected():
    with pytest.raises(ValueError):
        enumerate_tickets([four_favorites()[0]]*2,RiskThreshold())
