"""Parlay research must use same-book, chronological quotes and distinct games."""
import numpy as np
import pandas as pd
from nfl_oracle.parlays import compounded_decimal, decimal_to_american, parlay_watchlist

NOW=pd.Timestamp("2026-10-18T16:00:00Z")

def quote(game,book="Book A",updated="2026-10-18T15:45:00Z",model=.9):
    return dict(game_id=game,home_team="PHI",away_team="NYG",
                bookmaker=book,captured_at=updated,
                home_moneyline=-150,away_moneyline=130,
                model_home_prob=model)

def test_parlay_pricing_matches_compounded_leg_odds():
    assert np.isclose(compounded_decimal(-300,-200),2.)
    assert np.isclose(decimal_to_american(compounded_decimal(-300,-200)),100.)

def test_two_distinct_rejected_games_same_book_have_explicit_stake():
    q=pd.DataFrame([quote("a"),quote("b")])
    pick,audit=parlay_watchlist(q,pd.DataFrame({"game_id":[]}),NOW)
    assert len(pick)==1
    assert len(audit)==1
    x=pick.iloc[0]
    assert {x.game_a,x.game_b}=={"a","b"}
    assert x.bookmaker=="Book A"
    assert x.calculated_decimal_odds>2.
    assert x.estimated_ev_assuming_independence>0
    assert 2<=x.suggested_extra_stake<=15
    assert "EXPERIMENTAL" in x.research_status

def test_block_straight_games_and_cross_book_pairs():
    q=pd.DataFrame([quote("a"),quote("b"),quote("c","Book B")])
    pick,audit=parlay_watchlist(q,pd.DataFrame({"game_id":["a"]}),NOW)
    assert pick.empty and audit.empty

def test_incompatible_or_future_quotes_cannot_be_compounded():
    q=pd.DataFrame([quote("a"),quote("b",updated="2026-10-18T14:00:00Z")])
    pick,_=parlay_watchlist(q,pd.DataFrame({"game_id":[]}),NOW)
    assert pick.empty
    q.loc[1,"captured_at"]="2026-10-18T17:00:00Z"
    pick,_=parlay_watchlist(q,pd.DataFrame({"game_id":[]}),NOW)
    assert pick.empty

def test_negative_ev_remains_watch_only_and_zero_stake():
    q=pd.DataFrame([quote("a",model=.70),quote("b",model=.70)])
    q["home_moneyline"]=-400
    q["away_moneyline"]=300
    pick,_=parlay_watchlist(q,pd.DataFrame({"game_id":[]}),NOW)
    assert len(pick)==1
    assert pick.iloc[0].suggested_extra_stake==0
    assert pick.iloc[0].research_status=="WATCH_ONLY_UNVERIFIED_EDGE"
