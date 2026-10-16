"""Future-game same-book ML/ATS preview never touches unplayed outcomes."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from scripts.parlay_live_mixed import (
    kickoff_utc,parse_asof,extract_same_book_markets,
    check_quotes,build_preview,
)
from scripts.parlay_mixed_market_v5 import MixedPolicy


def upcoming_schedule():
    return pd.DataFrame([
        dict(game_id="2026_06_ARI_LA",home_team="LA",away_team="ARI",
             gameday="2026-10-18",gametime="16:05",season=2026,week=6,game_type="REG"),
        dict(game_id="2026_06_LAC_KC",home_team="KC",away_team="LAC",
             gameday="2026-10-18",gametime="16:25",season=2026,week=6,game_type="REG"),
    ])


def event(home,away,mlh,mla,point,game_day="2026-10-18T20:05:00Z"):
    return dict(home_team=home,away_team=away,commence_time=game_day,bookmakers=[
       dict(key="testbook",markets=[
         dict(key="h2h",last_update="2026-10-16T19:30:00Z",
              outcomes=[dict(name=home,price=mlh),dict(name=away,price=mla)]),
         dict(key="spreads",last_update="2026-10-16T19:31:00Z",
              outcomes=[dict(name=home,price=-110,point=point),
                        dict(name=away,price=-110,point=-point)])])
    ])


def test_rams_chiefs_market_quotes_preserved():
    games=upcoming_schedule()
    data=[
        event("Los Angeles Rams","Arizona Cardinals",-850,590,-12.5),
        event("Kansas City Chiefs","Los Angeles Chargers",-390,310,-7.5,
              "2026-10-18T20:25:00Z"),
    ]
    prices=extract_same_book_markets(data,games)
    assert len(prices)==2
    assert set(prices.game_id)==set(games.game_id)
    assert prices.set_index("game_id").loc["2026_06_ARI_LA","spread_line"]==12.5
    assert prices.set_index("game_id").loc["2026_06_LAC_KC","spread_line"]==7.5


def test_quote_after_asof_or_after_kickoff_is_rejected():
    schedules=upcoming_schedule()
    schedules["kickoff_utc"]=kickoff_utc(schedules)
    quotes=extract_same_book_markets([
        event("Los Angeles Rams","Arizona Cardinals",-850,590,-12.5),
    ],schedules)
    assert len(check_quotes(quotes,schedules,parse_asof(
        "2026-10-16T15:38:00-04:00")))==1
    assert check_quotes(quotes,schedules,parse_asof(
        "2026-10-16T15:29:00-04:00")).empty
    assert check_quotes(quotes,schedules,parse_asof(
        "2026-10-18T17:00:00-04:00")).empty


def test_invalid_asof_rejected():
    with pytest.raises(ValueError):
        parse_asof("2026-10-16T15:38:00")


def test_no_known_future_score_in_outcome_audit():
    from types import SimpleNamespace
    from scripts.parlay_mixed_market_v5 import analyze_game,ticket
    a=dict(game_id="2026_06_ARI_LA",season=2026,week=6,
        market_home_prob=.90,home_moneyline=-850,away_moneyline=590,
        spread_line=12.5,home_spread_odds=-110,away_spread_odds=-110,
        margin_prediction=13,true_margin=float("nan"))
    b=dict(game_id="2026_06_LAC_KC",season=2026,week=6,
        market_home_prob=.78,home_moneyline=-390,away_moneyline=310,
        spread_line=7.5,home_spread_odds=-110,away_spread_odds=-110,
        margin_prediction=8,true_margin=float("nan"))
    errors=np.array([-8,-5,-3,0,3,5,8.]*60)
    rule=MixedPolicy()
    ml_a=next(x for x in analyze_game(SimpleNamespace(**a),errors,
      "market_home_prob",rule) if x["kind"]=="ML" and x["side"]=="home")
    ml_b=next(x for x in analyze_game(SimpleNamespace(**b),errors,
      "market_home_prob",rule) if x["kind"]=="ML" and x["side"]=="home")
    result=ticket((ml_a,ml_b),rule,"market_home_prob")
    assert result["true_unit_return"] is None
    assert result["status"]!="QUALIFIES_RESEARCH"
    assert "2026_06_ARI_LA" in result["game_ids"]
    assert "2026_06_LAC_KC" in result["game_ids"]
