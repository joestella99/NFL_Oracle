import pandas as pd
from scripts.spread_comparison import evaluate_spread
from scripts.consensus_allocator import assemble
from scripts.weighted_slate import distribute

def test_home_spread_line_uses_positive_home_favorite_convention():
    # Home favored by 6.5; win by three does NOT cover.
    rows=[]
    for model in ('baseline','matchup','sustainability'):
        rows.append(dict(game_id='game',season=2024,week=1,model=model,prediction=.75,
          market_home_prob=.74,home_win=1,home_moneyline=-300,away_moneyline=240,
          home_spread_odds=-110,away_spread_odds=-110,spread_line=6.5,
          home_score=24,away_score=21))
    report,detail,_=evaluate_spread(pd.DataFrame(rows))
    assert report['moneyline_vs_spread']['moneyline']['profit']>0
    assert report['moneyline_vs_spread']['spread']['profit']<0
    assert detail.iloc[0].home_cover_margin==-3.5

def test_push_returns_stake():
    rows=[]
    for model in ('baseline','matchup','sustainability'):
        rows.append(dict(game_id='game',season=2024,week=1,model=model,prediction=.75,
          market_home_prob=.74,home_win=1,home_moneyline=-300,away_moneyline=240,
          home_spread_odds=-110,away_spread_odds=-110,spread_line=3.,
          home_score=24,away_score=21))
    report,detail,_=evaluate_spread(pd.DataFrame(rows))
    assert report['moneyline_vs_spread']['spread']['pushes']==1
    assert report['moneyline_vs_spread']['spread']['profit']==0
