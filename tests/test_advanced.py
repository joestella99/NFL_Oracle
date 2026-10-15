import pandas as pd
from nfl_oracle.advanced import aggregate_pbp, make_pregame_advanced

def test_aggregates_and_no_same_day_lookahead():
    games=pd.DataFrame([
      {'game_id':'g1','gameday':'2026-09-01','home_team':'A','away_team':'B'},
      {'game_id':'g2','gameday':'2026-09-01','home_team':'A','away_team':'C'},
      {'game_id':'g3','gameday':'2026-09-08','home_team':'A','away_team':'B'}])
    pbp=pd.DataFrame([
      {'game_id':'g1','posteam':'A','defteam':'B','epa':0.5,'play_type':'pass','passer_player_id':'qbA'},
      {'game_id':'g1','posteam':'A','defteam':'B','epa':0.1,'play_type':'pass','passer_player_id':'qbA'},
      {'game_id':'g1','posteam':'B','defteam':'A','epa':-0.3,'play_type':'run','passer_player_id':None},
      {'game_id':'g2','posteam':'A','defteam':'C','epa':1.0,'play_type':'pass','passer_player_id':'qbA'},
      {'game_id':'g2','posteam':'C','defteam':'A','epa':-0.2,'play_type':'run','passer_player_id':None}])
    agg=aggregate_pbp(pbp)
    assert round(float(agg.loc[(agg.game_id=='g1') & (agg.team=='A'),'off_epa'].iloc[0]),2)==0.3
    res=make_pregame_advanced(games,pbp).set_index('game_id')
    assert res.loc['g1','epa_offense_diff']==0
    assert res.loc['g2','epa_offense_diff']==0
    assert abs(res.loc['g3','epa_offense_diff'])>0

def test_unknown_pbp_empty():
    cols=['game_id','posteam','defteam','epa','play_type','passer_player_id']
    assert aggregate_pbp(pd.DataFrame(columns=cols)).empty
