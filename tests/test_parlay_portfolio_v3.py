"""Lock the explicitly chosen parlay V3 selection standards and risk accounting."""
from dataclasses import replace
import pytest

from scripts.parlay_portfolio_v2 import RiskThreshold, enumerate_tickets, allocate_tickets
from scripts.parlay_portfolio_v3 import load_policy

def four_favorites():
    favorites=[]
    for gid,p in zip("abcd", [.85,.82,.80,.78]):
        favorites.append(dict(
            game_id=gid,side="home",football_p=p,market_p=.69,
            conservative_p=p-.035,odds=-200,decimal=1.5,
            model_disagreement=.01,model_ev=p*1.5-1,
            conservative_leg_ev=(p-.035)*1.5*.98-1,
        ))
    return favorites



def test_official_policy_configuration_is_exactly_the_promoted_variant():
    descriptor, risk = load_policy()
    assert descriptor["status"] == "official_default_parlay_selection_policy"
    assert risk.min_conservative_ev == pytest.approx(.02)
    assert risk.per_leg_price_haircut == pytest.approx(.01)
    assert risk.require_positive_leg_ev is False
    assert risk.min_joint_probability == pytest.approx(.15)
    assert risk.maximum_game_risk_fraction == pytest.approx(.02)
    assert risk.maximum_total_game_risk_fraction == pytest.approx(.12)


def test_weak_individual_leg_is_recorded_not_automatically_vetoed():
    _,risk = load_policy()
    legs=four_favorites()[:2]
    legs[0]["conservative_leg_ev"] = -.01
    a=enumerate_tickets(legs,risk)
    b=enumerate_tickets(legs,replace(risk,require_positive_leg_ev=True))
    assert a.iloc[0].weak_leg_count == 1
    assert a.iloc[0].status == "ELIGIBLE_RESEARCH"
    assert b.iloc[0].status == "LEG_NOT_VALUE"


def test_parlay_exposure_can_overlap_with_preexisting_straight():
    _,risk = load_policy()
    tickets=enumerate_tickets(four_favorites(),risk)
    after=allocate_tickets(tickets,risk,{"a":50.})
    allocated=after[after.stake>0]
    assert not allocated.empty
    # Straight bet already consumes $50, but combined cap $120 permits some
    # additional parlay exposure; $20 separate parlay-game cap dominates.
    assert allocated.loc[allocated.game_ids.map(lambda g:"a" in g),"stake"].sum()>0
    assert allocated.loc[allocated.game_ids.map(lambda g:"a" in g),"stake"].sum()<=20.01
    assert allocated.stake.sum()<=50.01
    assert allocated.stake.max()<=5.01


def test_combined_cap_still_blocks_concentrated_game():
    _,risk=load_policy()
    tickets=enumerate_tickets(four_favorites(),risk)
    allocated=allocate_tickets(tickets,risk,{"a":120.})
    assert allocated.loc[allocated.game_ids.map(lambda g:"a" in g),"stake"].sum()==0
    assert allocated.stake.sum()>0


def test_four_heavy_favorites_remain_eleven_candidate_tickets():
    _,risk=load_policy()
    candidates=enumerate_tickets(four_favorites(),risk)
    assert len(candidates)==11
    assert set(candidates.status)=={"ELIGIBLE_RESEARCH"}
    assert candidates.legs.value_counts().to_dict()=={2:6,3:4,4:1}


def test_no_budget_spending_for_tickets_failing_ev():
    _,risk=load_policy()
    poor=replace(risk,min_conservative_ev=.95)
    decisions=allocate_tickets(enumerate_tickets(four_favorites(),poor),poor)
    assert decisions.stake.sum()==0


def test_legacy_v2_defaults_are_not_mutated():
    old=RiskThreshold()
    assert old.require_positive_leg_ev is True
    assert old.maximum_total_game_risk_fraction is None
    assert old.per_leg_price_haircut==pytest.approx(.02)
