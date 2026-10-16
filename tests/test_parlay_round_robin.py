"""All-legs round-robin exhaustive enumeration, correlated exposure and sizing."""
import numpy as np
import pandas as pd
from scripts.parlay_round_robin import (
    generate_round_robins, allocate_round_robins, simulate,
    _size_sleeve, MIN_JOINT_WIN_PROB, audit_correlated_exposure,
)

def candidates(n=5,season=2025,week=1,policy="risk_tolerant"):
    entries=[]
    for j in range(n):
        for side in ("home","away"):
            prob=.77 if side=="home" else .23
            entries.append({
                "season":season,"week":week,"policy":policy,
                "game_id":f"game{j}","kind":"ML","side":side,
                "win_prob":prob,"odds":-400 if side=="home" else 250,
                "won":True,"payout":1/3 if side=="home" else 2.5,
            })
    return pd.DataFrame(entries)

def test_exhaustive_lengths_and_reused_legs():
    combos,considered,nlegs=generate_round_robins(candidates(5),min_joint_prob=.01)
    assert nlegs==5
    assert considered==26  # 2^5 - 1 - 5
    assert set(x["leg_count"] for x in combos)=={2,3,4,5}
    assert any(x["leg_count"]==5 for x in combos)
    assert all(len(set(x["leg_game_ids"]))==len(x["leg_game_ids"]) for x in combos)
    uses=sum("game0" in x["leg_game_ids"] for x in combos)
    assert uses>1  # round robin reuse is allowed

def test_adaptive_bankroll_and_sleeve_never_exceed_capital():
    fake=[{"conservative_ev":.20,"joint_probability":.48,
           "conservative_joint_probability":.44,"calculated_decimal_odds":3.5}]
    size=_size_sleeve(fake,1000,"positive_only")
    assert size>0 and size<=1000
    assert _size_sleeve(fake,2000,"positive_only")>size
    assert _size_sleeve(fake,1000,"exploratory")==5
    assert _size_sleeve([],1000,"positive_only")==0

def test_zero_positive_value_rejects_all_even_high_probability():
    combos,_,_=generate_round_robins(candidates(5),min_joint_prob=.01)
    assert all(x["conservative_ev"]<0 for x in combos)
    assert allocate_round_robins(combos,1000,"positive_only")==[]

def test_overlapping_straights_and_parlay_tickets_possible():
    universe=candidates(5)
    singles=pd.DataFrame([{"game_id":"game0","policy":"risk_tolerant",
                           "season":2025,"week":1}])
    tickets,weekly=simulate(universe,singles,bankroll=1000)
    assert len(weekly)==2
    assert int(weekly.max_legs_considered.max())>=2
    q=tickets.loc[tickets["mode"].eq("exploratory")]
    assert not q.empty
    assert any(q.overlaps_straight)
    assert q.stake.sum()<=5.001
    assert q.leg_count.max()>=2
    assert q.games.str.contains("game0").sum()>1
    assert q.games.str.contains("game1").sum()>1

def test_per_game_exposure_audit_counts_shared_tickets_once_per_leg():
    t=pd.DataFrame([
        dict(policy="risk_tolerant",season=2025,week=1,mode="exploratory",
             games="game0 | game1",sides="home | away",stake=1.),
        dict(policy="risk_tolerant",season=2025,week=1,mode="exploratory",
             games="game0 | game2",sides="home | home",stake=2.),
        dict(policy="risk_tolerant",season=2025,week=1,mode="exploratory",
             games="game1 | game2",sides="away | home",stake=3.),
    ])
    s=pd.DataFrame([dict(game_id="game0",season=2025,week=1,
        policy="risk_tolerant",side="home",stake=40.)])
    audit=audit_correlated_exposure(t,s)
    a=audit[audit.game_id.eq("game0")].iloc[0]
    assert a.parlay_tickets_containing_leg==2
    assert a.parlay_stake_exposed==3
    assert a.weekly_parlay_staked==6
    assert a.pct_parlay_budget_at_risk==.5
    assert a.parlay_pnl_if_leg_loses==-3
    assert a.straight_stake_on_same_game==40
    assert a.straight_side=="home"
    # Overlap means per-game exposure sums exceed the actual $6 risk sleeve.
    assert audit.parlay_stake_exposed.sum()==12
    assert audit.straight_stake_on_same_game.sum()==40

def test_positive_staking_matches_edge_to_same_ticket_payout():
    options=[
        dict(conservative_ev=.20,conservative_joint_probability=.20,
             calculated_decimal_odds=21.),
        dict(conservative_ev=.12,conservative_joint_probability=.60,
             calculated_decimal_odds=2.),
    ]
    # Candidate 2 has a larger EV-to-profit ratio despite lower absolute EV:
    # 25% Kelly on 0.12, not on 0.20/20.
    assert _size_sleeve(options,1000,"positive_only")==30
