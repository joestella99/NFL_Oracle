import numpy as np
import pandas as pd
import pytest
from nfl_oracle.core import ml_to_implied, odds_profit, no_vig_home, make_features, rank_edges, backtest


def _game(date, i, h='PHI', a='DAL', hs=27, ass=20, hml=-125, aml=115, season=2023):
    return {'game_id':str(i),'season':season,'game_type':'REG','gameday':date,
       'home_team':h,'away_team':a,'home_score':hs,'away_score':ass,
       'home_moneyline':hml,'away_moneyline':aml,'home_rest':7,'away_rest':7}


def test_odds_conversion():
    assert ml_to_implied(-110)==pytest.approx(110/210)
    assert ml_to_implied(150)==pytest.approx(100/250)
    assert odds_profit(20,-140)==pytest.approx(14.285714)
    assert odds_profit(20,150)==pytest.approx(30)
    assert no_vig_home(-110,-110)==pytest.approx(.5)
    for x in [0, -80, 80, float('nan')]:
        with pytest.raises(ValueError):ml_to_implied(x)


def test_no_same_game_information_in_features():
    first=_game('2023-09-01',1,hs=100,ass=0)
    second=_game('2023-09-08',2,hs=20,ass=21)
    features=make_features(pd.DataFrame([second,first]))
    assert features.iloc[0]['home_n']==0
    assert features.iloc[0]['form_points_for_diff']==0
    assert features.iloc[1]['home_n']==1
    assert features.iloc[1]['form_points_for_diff']==100.0


def test_future_outcome_not_used_without_completed_score():
    f=make_features(pd.DataFrame([_game('2025-10-10',1,hs=None,ass=None),_game('2025-10-20',2,hs=None,ass=None)]))
    assert f['completed'].eq(False).all()
    assert f['home_n'].eq(0).all()


def test_market_and_no_bet():
    f=pd.DataFrame([{'game_id':'x','gameday':'2024-10-01','home_team':'PHI','away_team':'DAL',
                      'home_moneyline':-110,'away_moneyline':-110,'model_home_prob':.5}])
    assert rank_edges(f).empty
    f.loc[0,'model_home_prob']=0.65
    picks=rank_edges(f)
    assert len(picks)==1
    assert picks.iloc[0]['selection']=='PHI'
    assert picks.iloc[0]['expected_profit_per_1_dollar']>0


def test_backtest_held_out_season_reality_check():
    rng=np.random.default_rng(8)
    rows=[]
    teams=['PHI','DAL','NYG','WAS','KC','DEN','BUF','MIA']
    index=0
    for year in [2022,2023,2024,2025]:
        for wk in range(1,19):
            rng.shuffle(teams)
            for j in range(4):
                hs=int(rng.integers(6,48)); ass=int(rng.integers(6,48))
                rows.append(_game(f'{year}-09-01',index,h=teams[j],a=teams[7-j],hs=hs,ass=ass,season=year))
                index+=1
        # Dates same within year intentionally for the synthetic fixture only; test chronology uses order by ID.
    report,preds=backtest(pd.DataFrame(rows),test_year=2025,min_history=3)
    assert report['test_year']==2025
    assert 0<report['test_games']<=72
    assert 0<=report['model']['brier']<=1
    assert (preds['season']==2025).all()


def test_odds_api_event_mapping_is_strict():
    from nfl_oracle.odds import extract_quotes
    schedule=pd.DataFrame([_game('2026-10-18','2026_07_CAR_PHI',h='PHI',a='CAR',hs=None,ass=None,season=2026)])
    events=[{'home_team':'Philadelphia Eagles','away_team':'Carolina Panthers',
       'commence_time':'2026-10-18T17:00:00Z',
       'bookmakers':[{'title':'Test Book','last_update':'2026-10-18T15:00:00Z',
          'markets':[{'key':'h2h','outcomes':[
            {'name':'Philadelphia Eagles','price':-200},
            {'name':'Carolina Panthers','price':+160}]}]}]}]
    quotes=extract_quotes(events,schedule)
    assert len(quotes)==1
    assert quotes.iloc[0]['game_id']=='2026_07_CAR_PHI'
    assert quotes.iloc[0]['home_moneyline']==-200
    events[0]['away_team']='New York Giants'
    assert extract_quotes(events,schedule).empty
