"""ML/ATS mixed parlay probabilities and spread pushes."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from scripts.parlay_mixed_market_v5 import (
    MixedPolicy,load_policy,empirical_cover,analyze_game,
    ticket,build_slates,probability_diagnostics,
)


def test_official_parameters_and_no_stakes():
    _,policy=load_policy()
    assert policy.min_ticket_conservative_ev==.02
    assert policy.per_leg_price_haircut==.01
    assert policy.max_legs==4
    assert "bankroll" not in vars(policy) and "stake" not in vars(policy)


def game(n,spread=7.5,ml=-250,away_ml=200):
    return dict(game_id=f"2026_05_T{n}_O{n}",season=2026,week=5,
        market_home_prob=.70,football_base_prob=.70,
        football_injury_prob=.70,home_win=1,
        home_moneyline=ml,away_moneyline=away_ml,
        spread_line=spread,home_spread_odds=-110,away_spread_odds=-110,
        margin_prediction=8, true_margin=8)


def test_empirical_discrete_cover_and_push_mass():
    errors=np.array([0.]*100+[1.]*100+[-1.]*100)
    h,a,p=empirical_cover(7.,7.,errors)
    assert h==pytest.approx(1/3)
    assert a==pytest.approx(1/3)
    assert p==pytest.approx(1/3)
    h,a,p=empirical_cover(7.,7.5,errors)
    assert p==0.


def test_market_sides_audited_and_selected_two_per_game():
    from types import SimpleNamespace
    r=analyze_game(SimpleNamespace(**game(1)),
        np.array([-3.,-1.,0.,1.,3.]*100),"market_home_prob",MixedPolicy())
    assert len(r)==4
    assert {(x["kind"],x["side"]) for x in r}=={
        ("ML","home"),("ML","away"),("ATS","home"),("ATS","away")}
    chosen=[x for x in r if x["candidate_selected"]]
    assert len(chosen)==2
    assert {x["kind"] for x in chosen}=={"ATS","ML"}
    assert next(x for x in chosen if x["kind"]=="ML")["side"]=="home"


def test_prohibit_same_game_ml_and_spread():
    from types import SimpleNamespace
    r=analyze_game(SimpleNamespace(**game(1)),
        np.array([-1.,0.,1.]*100),"market_home_prob",MixedPolicy())
    with pytest.raises(ValueError,match="Same game"):
        ticket(tuple(x for x in r if x["candidate_selected"]),MixedPolicy(),"market_home_prob")


def test_parlay_spread_push_removes_just_that_leg():
    from types import SimpleNamespace
    s=analyze_game(SimpleNamespace(**game(1,spread=8)),
        np.array([-1.,0.,1.]*100),"market_home_prob",MixedPolicy())
    second=analyze_game(SimpleNamespace(**game(2,spread=7.5)),
        np.array([-1.,0.,1.]*100),"market_home_prob",MixedPolicy())
    ml=next(x for x in second if x["kind"]=="ML" and x["side"]=="home")
    spread=next(x for x in s if x["kind"]=="ATS" and x["side"]=="home")
    assert spread["pushed"] and ml["won"]
    tk=ticket((spread,ml),MixedPolicy(),"market_home_prob")
    assert tk["true_unit_return"]==pytest.approx(ml["odds_decimal"]-1)
    assert tk["contains_push"]


def test_four_game_two_market_cross_product():
    forecasts=pd.DataFrame([{
        k:v for k,v in game(i).items()
        if k not in ["spread_line","home_spread_odds","away_spread_odds",
                     "margin_prediction","true_margin"]
        } for i in range(4)])
    margin=pd.DataFrame([{
        k:v for k,v in game(i).items()
        if k in ["game_id","season","week","spread_line","home_spread_odds",
                 "away_spread_odds","margin_prediction","true_margin"]
        } for i in range(4)])
    past=pd.DataFrame(dict(season=[2025]*300,true_margin=[7.]*300,
        margin_prediction=[8.]*300))
    leg,tk,weekly=build_slates(forecasts,margin,past,MixedPolicy(),
                              variants=("market_home_prob",))
    assert len(leg)==16
    assert len(weekly)==1
    # C(4,2)*2**2 + C(4,3)*2**3 + C(4,4)*2**4 = 72
    assert len(tk)==72
    assert set(tk.legs)=={2,3,4}
    assert {"ML + ML","ML + ATS","ATS + ML","ATS + ATS"}.issubset(set(tk.market_mix))
    assert all(len(set(x.split(" | ")))==len(x.split(" | ")) for x in tk.game_ids)
    assert not any("stake" in col or "bankroll" in col for col in tk.columns)


def test_probability_threshold_can_reject_even_winning_parlay():
    from types import SimpleNamespace
    model=MixedPolicy(min_ticket_conservative_ev=.9)
    a=analyze_game(SimpleNamespace(**game(1)),
        np.array([-1.,0.,1.]*100),"market_home_prob",model)
    b=analyze_game(SimpleNamespace(**game(2)),
        np.array([-1.,0.,1.]*100),"market_home_prob",model)
    ml1=next(x for x in a if x["candidate_selected"] and x["kind"]=="ML")
    ml2=next(x for x in b if x["candidate_selected"] and x["kind"]=="ML")
    tk=ticket((ml1,ml2),model,"market_home_prob")
    assert tk["status"]!="QUALIFIES_RESEARCH"
