"""Historical relocation and future-game source validation tests."""
import pandas as pd
from nfl_oracle.core import make_features
from nfl_oracle.advanced import make_pregame_advanced
from nfl_oracle.matchups import matchup_features
from nfl_oracle.stability import make_stability
from nfl_oracle.team_identity import franchise_key


def fixture():
    games=pd.DataFrame([
        dict(game_id='one',season=2015,gameday='2015-11-01',
             home_team='STL',away_team='DAL',home_score=30,away_score=10,
             home_moneyline=-130,away_moneyline=120),
        dict(game_id='two',season=2016,gameday='2016-11-01',
             home_team='LA',away_team='DAL',home_score=None,away_score=None,
             home_moneyline=-120,away_moneyline=110),
        dict(game_id='three',season=2016,gameday='2016-11-08',
             home_team='LA',away_team='DAL',home_score=None,away_score=None,
             home_moneyline=-125,away_moneyline=115)])
    pbp=pd.DataFrame([
        dict(game_id='one',posteam='STL',defteam='DAL',epa=0.8,
             play_type='pass',passer_player_id='ramsqb'),
        dict(game_id='one',posteam='DAL',defteam='STL',epa=-0.4,
             play_type='pass',passer_player_id='dalqb'),
        # A still-open game can have live PBP. It must not update any next
        # game's pregame signals unless the schedule shows final scores.
        dict(game_id='two',posteam='LA',defteam='DAL',epa=-90.0,
             play_type='pass',passer_player_id='ramsqb'),
        dict(game_id='two',posteam='DAL',defteam='LA',epa=99.0,
             play_type='pass',passer_player_id='dalqb')])
    return games,pbp


def test_franchise_aliases_keep_rival_clubs_separate():
    assert franchise_key('STL')==franchise_key('LA')
    assert franchise_key('SD')==franchise_key('LAC')
    assert franchise_key('OAK')==franchise_key('LV')
    assert franchise_key('WSH')==franchise_key('WAS')
    assert franchise_key('LA')!=franchise_key('LAC')
    assert franchise_key('XYZ')=='XYZ'


def test_form_history_survives_relocation_without_renaming_game_id():
    games,_=fixture()
    features=make_features(games).set_index('game_id')
    assert features.loc['two','home_team']=='LA'
    assert features.loc['one','home_team']=='STL'
    assert features.loc['two','home_n']==1
    assert features.loc['two','form_margin_diff']==40
    assert features.loc['three','home_n']==1


def test_football_history_joins_relocated_pbp_and_excludes_live_game():
    games,pbp=fixture()
    advanced=make_pregame_advanced(games,pbp).set_index('game_id')
    matchups=matchup_features(games,pbp).set_index('game_id')
    stability=make_stability(games,pbp).set_index('game_id')
    assert advanced.loc['one','epa_offense_diff']==0
    assert round(advanced.loc['two','epa_offense_diff'],6)==1.2
    assert round(advanced.loc['three','epa_offense_diff'],6)==1.2
    assert round(matchups.loc['two','pass_attack_edge'],6)==1.2
    assert round(matchups.loc['three','pass_attack_edge'],6)==1.2
    assert stability.loc['two','success_rate_diff']==1
    assert stability.loc['three','success_rate_diff']==1


def test_unfinished_game_pbp_changes_never_alter_pregame_history():
    games,pbp=fixture()
    original=make_pregame_advanced(games,pbp).set_index('game_id')
    pbp.loc[pbp.game_id.eq('two'),'epa']=[+700.0,-700.0]
    changed=make_pregame_advanced(games,pbp).set_index('game_id')
    pd.testing.assert_frame_equal(original,changed)
