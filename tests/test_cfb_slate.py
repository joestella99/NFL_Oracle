import datetime as dt
from zoneinfo import ZoneInfo
from scripts.cfb_slate import build, decimal_odds

ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2026, 10, 24, 12, 6, tzinfo=ET)
BASE = {"game_id":"a", "kickoff_et":"2026-10-24T15:30:00-04:00",
        "away":"A","home":"B","market":"ml","selection":"A ML","american_odds":"+150",
        "model_probability":"","probability_se":""}

def test_price_math():
    assert decimal_odds("+150") == 2.5
    assert decimal_odds("-200") == 1.5

def test_missing_model_is_never_a_pick():
    report = build([BASE], NOW)
    assert report["singles"][0]["status"] == "UNSCORED_NO_CFB_MODEL"
    assert report["parlays"] == []

def test_started_game_cannot_qualify():
    game = dict(BASE, kickoff_et="2026-10-24T12:00:00-04:00",
                model_probability="0.80",probability_se="0.01")
    assert build([game], NOW)["singles"][0]["status"] == "STARTED"

def test_same_game_never_combined():
    a = dict(BASE,model_probability="0.7",probability_se="0.01")
    b = dict(a,selection="A +3", market="spread")
    c = dict(a,game_id="b",selection="C ML")
    report = build([a,b,c],NOW)
    assert report["parlays"]
    assert all(len(x["game_ids"]) == len(set(x["game_ids"])) for x in report["parlays"])
