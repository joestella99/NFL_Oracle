import pandas as pd
import pytest
from scripts.audit_margin_epa import tag_groups,evaluate

def sample(season):
    return pd.DataFrame({'game_id':[f'{season}-{i}' for i in range(8)],
      'season':[season]*8,'home_win':[0,1,0,1,0,1,1,0],
      'market_home_prob':[.3,.7,.4,.6,.4,.6,.7,.3],
      'form_margin_diff':[-4,4,-2,2,-3,3,1,-1],
      'epa_offense_diff':[.12,.13,-.2,.3,.1,-.15,.09,-.2]})

def test_groups_classify_hidden_team_epa():
    d=tag_groups(sample(2025))
    assert d.loc[0,'group']=='hidden_home_epa'
    assert d.loc[1,'group']=='both_home'
    assert d.loc[5,'group']=='hidden_away_epa'

def test_never_fit_on_same_season():
    with pytest.raises(ValueError,match='strictly'):
        evaluate(sample(2025),sample(2025))

def test_metrics_reproducible_on_future_season():
    r,s=evaluate(sample(2025),sample(2026))
    assert r['test_games']==8
    assert r['promotion']=='research_only'
    assert s.corrected_home_prob.between(0,1).all()
