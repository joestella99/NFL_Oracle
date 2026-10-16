"""Selection rules use pregame evidence and never fund no-edge parlays."""
import pandas as pd
from scripts.selective_parlays import select_portfolio

def fake(n=4, win=.82, odds=-120):
    rows=[]
    for i in range(n):
        rows.append(dict(game_id=f"g{i}",season=2025,week=1,
            policy="balanced",kind="ML",side="home",
            win_prob=win,odds=odds,won=True))
    return pd.DataFrame(rows)

def test_selective_funds_distinct_and_overlapping_positive_value():
    tickets,audit=select_portfolio(fake(4),1000)
    assert audit["examined"]==11
    assert audit["eligible_legs"]==4
    assert len(tickets)>0
    assert all(t["conservative_ev"]>.02 for t in tickets)
    assert all(t["stake"]>=1 for t in tickets)
    assert sum(t["stake"] for t in tickets)<=1000
    assert max(t["leg_count"] for t in tickets)<=4
    assert len({t["games"] for t in tickets})==len(tickets)

def test_no_parlays_when_probabilities_fail_value_check():
    tickets,audit=select_portfolio(fake(4,win=.65,odds=-300),1000)
    assert tickets==[]
    assert audit["funded"]==0
