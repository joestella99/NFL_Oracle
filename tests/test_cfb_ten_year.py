import pandas as pd
from types import SimpleNamespace
from scripts.cfb_ten_year import (
    decimal_odds, cdf, candidate, fit_prior, make_parlays, settle_parlay, summarize
)

class FakeLogistic:
    def predict_proba(self, X):
        return [[0.40, 0.60]]

class FakeRidge:
    def predict(self, X):
        return [7.0]

def test_prices():
    assert decimal_odds(150) == 2.5
    assert decimal_odds(-200) == 1.5

def test_probability():
    assert abs(cdf(0)-.5) < 1e-12

def test_spread_cover_and_pushing_price_assumption():
    g=SimpleNamespace(game_id=1,season=2024,week=2,
          home_team="Home",away_team="Away",
          elo_delta=0.3,home_field=1,home_margin=7)
    m=SimpleNamespace(market_type="spread",book="PINNACLE",
          lines_home=-7.0,lines_away=7.0,odds_home=float("nan"),odds_away=float("nan"))
    picks=candidate(g,m,FakeLogistic(),FakeRidge(),17)
    assert len(picks)==2
    assert picks[0]["push"] and picks[1]["push"]
    assert picks[0]["unit_profit"]==0
    assert picks[0]["assumed_spread_price"]

def test_ml_keeps_true_recorded_prices():
    g=SimpleNamespace(game_id=2,season=2024,week=2,
          home_team="Home",away_team="Away",
          elo_delta=0.3,home_field=1,home_margin=-7)
    m=SimpleNamespace(market_type="money_line",book="PINNACLE",
          lines_home=float("nan"),lines_away=float("nan"),odds_home=-150,odds_away=130)
    picks=candidate(g,m,FakeLogistic(),FakeRidge(),17)
    assert len(picks)==2
    assert not picks[0]["won"] and picks[1]["won"]
    assert abs(picks[1]["unit_profit"]-1.3)<1e-6
    assert not picks[0]["assumed_spread_price"]

def test_parlay_lose_overrides_push():
    a={"won":True,"push":False,"american_odds":150}
    b={"won":False,"push":False,"american_odds":-110}
    c={"won":False,"push":True,"american_odds":-110}
    assert settle_parlay([a,b,c])==-1
    assert settle_parlay([a,c])==1.5

def test_summary_no_bets():
    assert summarize(pd.DataFrame())["roi"] is None

def test_walk_forward_excludes_target_season():
    training=pd.DataFrame({
        "season":[2014,2014,2015,2015,2016,2016],
        "elo_delta":[-1,1,-.8,.8,-1,1],
        "home_field":[1,1,0,1,1,1],
        "home_win":[0,1,0,1,1,0],
        "home_margin":[-14,14,-7,7,1000,-1000],
    })
    a=fit_prior(training,2016)
    b=fit_prior(training[training.season<2016],2016)
    assert abs(a[1].predict([[.1,1]])[0] - b[1].predict([[.1,1]])[0]) < 1e-12
    assert a[2]==b[2]

def test_parlay_one_per_week():
    rows=[]
    for i in range(4):
        rows.append({"season":2025,"week":3,"game_id":i+1,
            "conservative_ev":.07,"stressed_probability":.68,
            "american_odds":120,"selection":str(i),"won":True,
            "push":False,"assumed_spread_price":False})
    p=make_parlays(pd.DataFrame(rows),2)
    assert len(p)==1
    assert len(p.iloc[0].game_ids.split(" | "))==2
