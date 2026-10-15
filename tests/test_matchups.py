import pandas as pd
from nfl_oracle.matchups import matchup_features, MATCHUP_FEATURES

def fixtures():
    games=pd.DataFrame([
    dict(game_id='g1',gameday='2025-09-07',home_team='A',away_team='B',home_score=24,away_score=10),
    dict(game_id='g2',gameday='2025-09-07',home_team='C',away_team='D',home_score=3,away_score=20),
    dict(game_id='g3',gameday='2025-09-14',home_team='A',away_team='C',home_score=10,away_score=20)])
    plays=pd.DataFrame([
    dict(game_id='g1',posteam='A',defteam='B',play_type='pass',epa=0.5),
    dict(game_id='g1',posteam='B',defteam='A',play_type='pass',epa=-0.3),
    dict(game_id='g2',posteam='C',defteam='D',play_type='run',epa=0.2),
    dict(game_id='g2',posteam='D',defteam='C',play_type='pass',epa=0.1),
    dict(game_id='g3',posteam='A',defteam='C',play_type='pass',epa=20.0)])
    return games,plays

def test_no_same_day_leakage():
    games,plays=fixtures()
    actual=matchup_features(games,plays)
    assert len(actual)==3
    assert actual[MATCHUP_FEATURES].notna().all().all()
    assert (actual.set_index('game_id').loc[['g1','g2'],MATCHUP_FEATURES]==0).all().all()
    assert actual.set_index('game_id').loc['g3','pass_attack_edge'] == 0.5

def test_future_pbp_cannot_change_past_features():
    games,plays=fixtures()
    a=matchup_features(games,plays)
    plays.loc[plays.game_id.eq('g3'),'epa']=-99
    b=matchup_features(games,plays)
    pd.testing.assert_frame_equal(a,b)

def test_no_completed_game_no_history_update():
    games,plays=fixtures()
    games.loc[games.game_id=='g1',['home_score','away_score']]=float('nan')
    a=matchup_features(games,plays).set_index('game_id')
    assert a.loc['g3','pass_attack_edge']==0.
