"""Chronological NFL win probabilities: no future games in team form features."""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORM_WINDOW = 8
BASE_POINTS = 22.0
FEATURES = [
    'market_logit', 'form_points_for_diff', 'form_points_against_diff',
    'form_margin_diff', 'form_win_pct_diff', 'form_total_diff',
    'form_games_diff', 'rest_diff',
]
REQUIRED = ['game_id', 'season', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score',
            'home_moneyline', 'away_moneyline']


def ml_to_implied(odds: float) -> float:
    """Bookmaker implied probability, before overround normalization."""
    odds = float(odds)
    if not math.isfinite(odds) or odds == 0 or -100 < odds < 100:
        raise ValueError('American odds must be <= -100 or >= +100')
    return (100 / (odds + 100)) if odds > 0 else (-odds / (-odds + 100))


def odds_profit(stake: float, odds: float) -> float:
    """Net gain from a winning American-odds wager, excluding returned stake."""
    odds = float(odds)
    ml_to_implied(odds)
    return stake * ((odds / 100) if odds > 0 else (100 / -odds))


def no_vig_home(home_ml: float, away_ml: float) -> float:
    home, away = ml_to_implied(home_ml), ml_to_implied(away_ml)
    return home / (home + away)


def _mean(stats: deque[tuple[float, float, float]], i: int, default: float) -> float:
    return float(np.mean([s[i] for s in stats])) if stats else default


def make_features(games: pd.DataFrame) -> pd.DataFrame:
    """One pregame row per game. Completed games update form AFTER their row.

    Important: historical moneylines from nflverse are recorded lines, not
    time-stamped execution quotes. Real ROI requires time-stamped book snapshots.
    """
    missing = [x for x in REQUIRED if x not in games.columns]
    if missing:
        raise ValueError(f'Missing required schedule columns: {missing}')
    g = games.copy()
    g['gameday'] = pd.to_datetime(g['gameday'], errors='raise')
    g = g.sort_values(['gameday', 'game_id']).drop_duplicates('game_id')
    history = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    rows = []
    for _, game in g.iterrows():
        home, away = str(game['home_team']), str(game['away_team'])
        h, a = history[home], history[away]
        def get_form(d):
            pf = _mean(d, 0, BASE_POINTS)
            pa = _mean(d, 1, BASE_POINTS)
            wins = _mean(d, 2, 0.5)
            return pf, pa, pf-pa, wins, pf+pa
        hf, af = get_form(h), get_form(a)
        def valid_ml(name):
            try: return ml_to_implied(game[name])
            except (ValueError, TypeError, OverflowError): return None
        hml = valid_ml('home_moneyline')
        aml = valid_ml('away_moneyline')
        market = hml/(hml+aml) if hml is not None and aml is not None else np.nan
        market_logit = math.log(market/(1-market)) if np.isfinite(market) else np.nan
        hr = pd.to_numeric(game.get('home_rest', 7), errors='coerce')
        ar = pd.to_numeric(game.get('away_rest', 7), errors='coerce')
        hr = 7 if pd.isna(hr) else float(hr)
        ar = 7 if pd.isna(ar) else float(ar)
        row = {
            'game_id': game['game_id'], 'season': int(game['season']), 'gameday': game['gameday'],
            'home_team': home, 'away_team': away,
            'home_moneyline': game['home_moneyline'], 'away_moneyline': game['away_moneyline'],
            'market_home_prob': market, 'market_logit': market_logit,
            'form_points_for_diff': hf[0]-af[0],
            'form_points_against_diff': hf[1]-af[1],
            'form_margin_diff': hf[2]-af[2],
            'form_win_pct_diff': hf[3]-af[3],
            'form_total_diff': hf[4]-af[4],
            'form_games_diff': len(h)-len(a),
            'rest_diff': float(np.clip(hr-ar, -14, 14)),
            'home_n': len(h), 'away_n': len(a),
        }
        hs = pd.to_numeric(game['home_score'], errors='coerce')
        aps = pd.to_numeric(game['away_score'], errors='coerce')
        completed = bool(pd.notna(hs) and pd.notna(aps))
        row['completed'] = completed
        row['tie'] = completed and hs == aps
        row['home_win'] = (int(hs > aps) if completed else np.nan)
        rows.append(row)
        if completed:
            history[home].append((float(hs), float(aps), float(hs>aps)))
            history[away].append((float(aps), float(hs), float(aps>hs)))
    return pd.DataFrame(rows)


def eligible(df: pd.DataFrame, min_history: int = 3) -> pd.DataFrame:
    return df.loc[
        df['completed'] & ~df['tie']
        & (df['home_n']>=min_history) & (df['away_n']>=min_history)
        & df[FEATURES].notna().all(axis=1)
    ].copy()


def fit_model(training: pd.DataFrame, sample_weight=None, feature_cols=None):
    if training['home_win'].nunique() < 2 or len(training) < 70:
        raise ValueError('Need at least 70 historically completed, two-class training games.')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1500))
    kwargs = {'logisticregression__sample_weight': sample_weight} if sample_weight is not None else {}
    model.fit(training[feature_cols or FEATURES], training['home_win'].astype(int), **kwargs)
    return model


def evaluate(predicted: pd.DataFrame, prob_col: str = 'model_home_prob') -> dict:
    y = predicted['home_win'].astype(int)
    p = np.clip(predicted[prob_col].astype(float), 1e-6, 1-1e-6)
    out = {
        'games': len(y), 'brier': float(brier_score_loss(y,p)),
        'log_loss': float(log_loss(y,p,labels=[0,1])),
        'accuracy_50': float(((p>=0.5)==y).mean()),
    }
    out['auc'] = float(roc_auc_score(y,p)) if y.nunique()==2 else None
    return out


def backtest(games: pd.DataFrame, test_year: int = 2025, min_history: int = 3) -> tuple[dict,pd.DataFrame]:
    """Fit only previous seasons; test on a held-out season, one row per game."""
    features = eligible(make_features(games), min_history=min_history)
    years = features['season'].astype(int)
    train = features.loc[years < test_year]
    test = features.loc[years == test_year].copy()
    if len(test)==0: raise ValueError(f'No eligible test games in {test_year}.')
    model = fit_model(train)
    test['model_home_prob'] = model.predict_proba(test[FEATURES])[:,1]
    result = {
        'train_games':len(train), 'test_games':len(test), 'test_year':test_year,
        'model':evaluate(test,'model_home_prob'),
        'market':evaluate(test,'market_home_prob'),
        'note':'Recorded historical prices lack verified pregame timestamps. ROI is only a proxy.',
    }
    # Each game gives two mutually exclusive sides. Only flag if high estimated EV.
    candidates = rank_edges(test, minimum_ev=0.05, minimum_prob_edge=0.05)
    chosen = candidates.drop_duplicates(subset=['game_id'],keep='first')
    if len(chosen):
        chosen = chosen.merge(test[['game_id','home_win']],on='game_id',how='left')
        chosen['correct'] = np.where(chosen['side']=='home',chosen['home_win']==1,chosen['home_win']==0)
        chosen['profit_one_unit'] = np.where(chosen['correct'], chosen['american_odds'].map(lambda x: odds_profit(1,x)),-1.0)
    result['candidate_bets'] = len(chosen)
    result['candidate_roi_proxy'] = float(chosen['profit_one_unit'].mean()) if len(chosen) else None
    return result,test


def rank_edges(games: pd.DataFrame, minimum_ev: float = 0.05, minimum_prob_edge: float = 0.05) -> pd.DataFrame:
    """Research shortlist, not verified executable quotes. Each match yields at most two possible sides."""
    rows=[]
    for _,g in games.iterrows():
        home_p=float(g['model_home_prob'])
        if not math.isfinite(home_p): continue
        for side, p, odds in [('home',home_p,g['home_moneyline']),('away',1-home_p,g['away_moneyline'])]:
            try:
                raw_prob=ml_to_implied(odds)
                ev=p*odds_profit(1,odds)-(1-p)
            except (ValueError, TypeError):continue
            edge=p-raw_prob
            if ev < minimum_ev or edge < minimum_prob_edge:continue
            rows.append({'game_id':g['game_id'], 'gameday':g['gameday'],
                'home_team':g['home_team'],'away_team':g['away_team'], 'side':side,
                'selection':g['home_team'] if side=='home' else g['away_team'],
                'american_odds':float(odds), 'model_probability':p,
                'sportsbook_implied_probability':raw_prob,'edge_vs_raw_market':edge,
                'expected_profit_per_1_dollar':ev})
    if not rows:
        return pd.DataFrame(columns=['game_id','gameday','selection','side','american_odds','model_probability','sportsbook_implied_probability','edge_vs_raw_market','expected_profit_per_1_dollar'])
    return pd.DataFrame(rows).sort_values('expected_profit_per_1_dollar',ascending=False).reset_index(drop=True)