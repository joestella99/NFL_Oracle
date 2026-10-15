"""Market-independent evaluation must never use sportsbook lines as features."""
import numpy as np
from scripts.replay_weeks import walk_forward_predictions
from tests.test_replay_weeks import synthetic_seasons


def test_market_blind_ignores_changes_in_odds():
    games = synthetic_seasons()
    first = walk_forward_predictions(games, 2026, [5], include_market=False).sort_values('game_id')
    changed = games.copy()
    changed['home_moneyline'] = -220
    changed['away_moneyline'] = 180
    second = walk_forward_predictions(changed, 2026, [5], include_market=False).sort_values('game_id')
    np.testing.assert_allclose(first['model_home_prob'], second['model_home_prob'])
    assert first['uses_market_feature'].eq(False).all()