import pandas as pd
from nfl_oracle.stability import make_stability,STABILITY_FEATURES

def test_future_plays_do_not_change_pregame_inputs():
    games=pd.DataFrame([
      {'game_id':'g1','gameday':'2025-09-07','home_team':'A','away_team':'B','home_score':20,'away_score':10},
      {'game_id':'g2','gameday':'2025-09-14','home_team':'A','away_team':'B','home_score':17,'away_score':14}])
    p=pd.DataFrame([
      {'game_id':'g1','posteam':'A','play_type':'pass','epa':.3},
      {'game_id':'g1','posteam':'B','play_type':'run','epa':-.2},
      {'game_id':'g2','posteam':'A','play_type':'pass','epa':2.0}])
    before=make_stability(games,p)
    assert len(before)==2
    assert before[STABILITY_FEATURES].notna().all().all()
    assert (before.loc[0,STABILITY_FEATURES]==0).all()
    p.loc[p.game_id=='g2','epa']=-200
    after=make_stability(games,p)
    pd.testing.assert_frame_equal(before,after)

def test_unfinished_games_do_not_enter_history():
    games=pd.DataFrame([
      {'game_id':'g1','gameday':'2025-09-07','home_team':'A','away_team':'B','home_score':None,'away_score':None},
      {'game_id':'g2','gameday':'2025-09-14','home_team':'A','away_team':'B','home_score':17,'away_score':14}])
    p=pd.DataFrame([{'game_id':'g1','posteam':'A','play_type':'pass','epa':2.0}])
    out=make_stability(games,p)
    assert (out[STABILITY_FEATURES]==0).all().all()
