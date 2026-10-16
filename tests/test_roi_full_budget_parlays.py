"""Regression checks for full weekly spend and cross-game parlay accounting."""
import numpy as np
import pandas as pd
from scripts.roi_full_budget_parlays import make_singles, parlay_from_rejected, distribute_cents, compounded_parlay_decimal

def row(game, kind, side, risk, prob, qualifies, week=1, policy="balanced"):
    payout=.25 if kind=="ML" else .9090909091
    return dict(season=2025, week=week, policy=policy, game_id=game,
                kind=kind, side=side, qualifies=qualifies,
                risk_score=risk, estimated_ev=risk+.01,
                win_prob=prob, payout=payout,
                odds=-400 if kind=="ML" else -110,
                won=True, push=False)

def test_full_100_with_variable_number_of_qualifiers():
    d=pd.DataFrame([
        row("a","ML","home",.04,.8,True),
        row("a","ATS","home",.03,.5,True),
        row("b","ATS","away",.02,.51,True),
        row("c","ML","home",.01,.81,True),
        row("d","ML","home",-.2,.75,False),
    ])
    selected=make_singles(d)
    assert len(selected)==3
    assert selected.game_id.nunique()==3
    assert selected.stake.sum()==100
    assert (selected.stake>0).all()
    assert not selected.forced_fallback.any()

def test_fallback_bets_all_100_on_highest_win_probability_ml():
    d=pd.DataFrame([
        row("a","ML","home",-.2,.8,False),
        row("b","ML","home",-.3,.92,False),
        row("c","ATS","home",-.4,.97,False),
    ])
    s=make_singles(d)
    assert len(s)==1 and s.iloc[0].game_id=="b"
    assert bool(s.iloc[0].forced_fallback)
    assert s.stake.sum()==100

def test_multiple_weeks_spend_full_weekly_amount():
    d=pd.DataFrame([row("a","ML","home",.1,.82,True,week=1),
                    row("b","ML","home",.02,.91,True,week=2)])
    s=make_singles(d)
    assert s.groupby(["season","week","policy"]).stake.sum().eq(100).all()

def test_parlay_different_games_no_overlap_and_extra_stake():
    eligible=pd.DataFrame([
        row("a","ATS","home",.05,.52,True),
        row("a","ML","home",-.03,.78,False),
        row("b","ML","home",-.03,.77,False),
        row("c","ML","home",-.03,.76,False),
        row("d","ML","home",-.04,.75,False),
    ])
    singles=make_singles(eligible)
    parlay,reviewed=parlay_from_rejected(eligible,singles)
    assert len(parlay)==1
    r=parlay.iloc[0]
    assert r.game_a!=r.game_b
    assert r.game_a!='a' and r.game_b!='a'
    assert 2<=r.stake<=15
    assert not r.historical_quote_executable
    assert np.isclose(r.calculated_parlay_decimal_odds,(1+.25)**2)
    assert not reviewed.empty

def test_no_manufactured_positive_parlay_value():
    d=pd.DataFrame([row("a","ML","home",-.05,.62,False),
                    row("b","ML","home",-.05,.62,False),
                    row("c","ATS","home",.1,.55,True)])
    s=make_singles(d)
    t,_=parlay_from_rejected(d,s)
    assert t.empty

def test_action_network_parlay_calculator_examples():
    # Action Network combines -300, -200 and -150 at approximately +233.
    assert np.isclose(compounded_parlay_decimal(-300,-200),2.0)
    assert np.isclose(compounded_parlay_decimal(-300,-200,-150),3.3333333333)
    assert np.isclose(compounded_parlay_decimal(-110,-110,-110), (1+100/110)**3)
