"""Unlimited weekly parlay lengths and round-robin reuse, priced by one sportsbook."""
import numpy as np
import pandas as pd
from nfl_oracle.parlays import multi_leg_parlay_watchlist

T=pd.Timestamp("2026-10-18T16:00:00Z")

def mk(game,odds=100,home=.99,book="Book A",stamp="2026-10-18T15:45:00Z"):
    return dict(game_id=game,home_team="PHI",away_team="NYG",bookmaker=book,
       captured_at=stamp,home_moneyline=odds,away_moneyline=-110,
       model_home_prob=home)

def test_unbounded_legs_round_robins_and_overlapping_straights():
    # Artificial prices used only for mechanical tests: high model p and +100
    # demonstrate that >2 leg round robins get generated and share risk pool.
    q=pd.DataFrame([mk("g"+str(i)) for i in range(5)])
    picks,audit=multi_leg_parlay_watchlist(q,pd.DataFrame({"game_id":["g0"]}),T,
        parlay_bankroll=1000,min_joint_probability=.01)
    assert set(audit.leg_count)=={2,3,4,5}
    assert len(audit)==26
    assert not picks.empty
    assert len(picks)>1
    assert picks.leg_count.max()==5
    assert picks.overlaps_straight.any()
    assert picks.suggested_extra_stake.sum()<=1000.005
    assert all(len(x.split(" | "))==len(set(x.split(" | "))) for x in picks.game_ids)

def test_no_cross_book_or_untimed_parlay():
    q=pd.DataFrame([mk("a"),mk("b",book="Book B"),mk("c",stamp="2026-10-18T14:00:00Z")])
    picks,audit=multi_leg_parlay_watchlist(q,pd.DataFrame(),T)
    assert picks.empty and audit.empty

def test_no_positive_value_does_not_invent_home_run_stakes():
    q=pd.DataFrame([mk("a",odds=-400,home=.72),mk("b",odds=-400,home=.72)])
    picks,audit=multi_leg_parlay_watchlist(q,pd.DataFrame(),T)
    assert picks.empty
    assert (audit.conservative_ev<=0).all()
