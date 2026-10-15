"""Offline tests for the held-out weekly confidence report."""
import pandas as pd
import pytest

from scripts.replay_weeks import build_report


def schedule():
    return pd.DataFrame([
        dict(game_id='g1', season=2026, week=1, game_type='REG', home_score=31, away_score=21,
             gameday='2026-09-10', home_team='A', away_team='B'),
        dict(game_id='g2', season=2026, week=1, game_type='REG', home_score=14, away_score=20,
             gameday='2026-09-11', home_team='C', away_team='D'),
        dict(game_id='g3', season=2026, week=2, game_type='REG', home_score=17, away_score=17,
             gameday='2026-09-17', home_team='E', away_team='F'),
        dict(game_id='g4', season=2026, week=2, game_type='REG', home_score=28, away_score=3,
             gameday='2026-09-18', home_team='G', away_team='H'),
        dict(game_id='g5', season=2026, week=2, game_type='REG', home_score=float('nan'), away_score=float('nan'),
             gameday='2026-09-18', home_team='I', away_team='J'),
        dict(game_id='g6', season=2026, week=2, game_type='REG', home_score=28, away_score=3,
             gameday='2026-09-18', home_team='K', away_team='L'),
    ])


def predicted():
    return pd.DataFrame([
        dict(game_id='g1', gameday='2026-09-10', home_team='A', away_team='B',
             model_home_prob=0.65, market_home_prob=0.70, home_win=1,
             home_moneyline=-200, away_moneyline=170),
        dict(game_id='g2', gameday='2026-09-11', home_team='C', away_team='D',
             model_home_prob=0.70, market_home_prob=0.65, home_win=0,
             home_moneyline=-160, away_moneyline=140),
        dict(game_id='g4', gameday='2026-09-18', home_team='G', away_team='H',
             model_home_prob=0.52, market_home_prob=0.45, home_win=1,
             home_moneyline=115, away_moneyline=-135),
    ])


def test_full_confidence_report_and_exclusions():
    report, picks = build_report(predicted(), schedule(), 2026, [1, 2])
    assert report['overall']['games'] == 3
    assert report['overall']['correct'] == 2
    assert report['overall']['accuracy'] == 0.666667
    assert report['overall']['average_confidence'] == pytest.approx((.65+.70+.52)/3)
    assert report['overall']['confidence_minus_accuracy'] == pytest.approx((.65+.70+.52)/3-2/3,abs=1e-6)
    assert report['overall']['mean_confidence_correct'] == pytest.approx((.65+.52)/2)
    assert report['overall']['mean_confidence_wrong'] == 0.70
    assert report['by_week'][0]['games'] == 2
    assert report['by_week'][1]['games'] == 1
    assert report['by_week'][1]['ties_excluded'] == 1
    assert report['by_week'][1]['unscored_excluded'] == 1
    assert report['by_week'][1]['no_prediction_excluded'] == 1
    assert picks['predicted_winner'].tolist() == ['A', 'C', 'G']
    assert picks['correct'].tolist() == [True, False, True]
    assert sum(row['games'] for row in report['by_confidence_bucket']) == 3


def test_no_eligible_games_rejected():
    with pytest.raises(ValueError, match='No completed'):
        build_report(predicted().iloc[:0], schedule(), 2026, [1, 2])


def test_probabilities_outside_zero_one_rejected():
    p = predicted()
    p.loc[0,'model_home_prob']=1.2
    with pytest.raises(ValueError, match='Invalid probability'):
        build_report(p, schedule(), 2026, [1])


def synthetic_seasons():
    """Small offline league with unique teams per week and realistic game IDs."""
    result = []
    teams = [f'T{i}' for i in range(16)]
    for season in [2024, 2025, 2026]:
        for week in range(1, 7):
            for pair in range(8):
                home = teams[pair] if (season + week + pair) % 2 else teams[15-pair]
                away = teams[15-pair] if (season + week + pair) % 2 else teams[pair]
                result.append(dict(
                    game_id=f'{season}_{week}_{pair}', season=season, week=week,
                    game_type='REG', gameday=f'{season}-09-{week + 1:02d}',
                    home_team=home, away_team=away,
                    home_score=18 + ((season + week + pair * 3) % 19),
                    away_score=10 + ((season + week * 3 + pair * 2) % 17) +
                               int(18 + ((season + week + pair * 3) % 19) == 10 + ((season + week * 3 + pair * 2) % 17)),
                    home_moneyline=-150, away_moneyline=130,
                    home_rest=7, away_rest=7,
                ))
    return pd.DataFrame(result)


def test_weekly_refit_only_uses_previous_weeks():
    from scripts.replay_weeks import walk_forward_predictions
    full = synthetic_seasons()
    results = walk_forward_predictions(full, 2026, [1,2,3,4,5])
    assert len(results) == 5 * 8
    assert results.groupby('week')['current_season_training_games'].first().tolist() == [0,8,16,24,32]
    assert results.groupby('week')['training_through_week'].first().tolist() == [0,1,2,3,4]
    assert (results['current_season_weight'] == 4.0).all()
    assert results['model_home_prob'].between(0,1).all()
    assert results['static_home_prob'].between(0,1).all()


def test_future_results_cannot_change_week_five_prediction():
    from scripts.replay_weeks import walk_forward_predictions
    games = synthetic_seasons()
    first = walk_forward_predictions(games, 2026, [5]).sort_values('game_id')
    changed = games.copy()
    mask = (changed['season'] == 2026) & (changed['week'] >= 5)
    changed.loc[mask,['home_score','away_score']] = changed.loc[mask,['away_score','home_score']].to_numpy()
    second = walk_forward_predictions(changed, 2026, [5]).sort_values('game_id')
    assert first['model_home_prob'].to_numpy() == pytest.approx(second['model_home_prob'].to_numpy())


def test_current_weight_must_be_valid():
    from scripts.replay_weeks import walk_forward_predictions
    with pytest.raises(ValueError, match='current_season_weight'):
        walk_forward_predictions(synthetic_seasons(), 2026, [5], current_season_weight=0.5)