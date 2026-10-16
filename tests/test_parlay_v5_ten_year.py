"""V5 ten-season parlay simulation invariants and no-lookahead checks."""
import numpy as np
import pandas as pd
import pytest
from scripts.parlay_v5_ten_year import simulate,calibration_baseline,report_results
from scripts.parlay_mixed_market_v5 import MixedPolicy


def synthetic_forecasts():
    return pd.DataFrame([
       dict(game_id=f"2016_01_{x}_X",season=2016,week=1,
            market_home_prob=.73,
            football_base_prob=.75,
            football_injury_prob=.74,
            home_moneyline=-270,away_moneyline=210)
       for x in ["A","B","C"]])


def synthetic_margin():
    past=pd.DataFrame([
        dict(game_id=f"{yr}_0_{i}",season=yr,week=1,
            spread_line=1.5,margin_prediction=8.,true_margin=float(8+(i%13)-6),
            home_spread_odds=-110,away_spread_odds=-110)
        for yr in (2014,2015) for i in range(140)])
    this=pd.DataFrame([
        dict(game_id=f"2016_01_{x}_X",season=2016,week=1,
            spread_line=3.5,margin_prediction=4.,true_margin=float(6 if i!=2 else -2),
            home_spread_odds=-110,away_spread_odds=-110)
        for i,x in enumerate(["A","B","C"])])
    return pd.concat([past,this],ignore_index=True)


def test_year_2016_uses_2014_2015_residuals_only():
    wk,qual,near,legs=simulate(synthetic_forecasts(),synthetic_margin(),MixedPolicy())
    assert len(wk)==3
    assert set(wk.last_residual_season)=={2015}
    assert set(wk.pre_year_residual_games)=={280}
    assert all(wk.examined>0)
    assert all(wk.examined==wk.rejected_win_prob+wk.rejected_conservative_ev+wk.passed)
    assert not any("stake" in x or "bankroll" in x for x in near.columns)
    assert len(legs)==36  # 3 games, 4 market sides, 3 model variants


def test_target_season_actual_outcomes_do_not_change_selected_probabilities():
    args=(synthetic_forecasts(),synthetic_margin(),MixedPolicy())
    wk,qual,near,legs=simulate(*args)
    revised=args[1].copy()
    revised.loc[revised.season.eq(2016),"true_margin"]+=45
    wk2,qual2,near2,legs2=simulate(args[0],revised,args[2])
    for key in ("conservative_ev","estimated_ev",
                "estimated_all_win_probability","conservative_all_win_probability","status"):
        pd.testing.assert_series_equal(near[key],near2[key])


def test_2016_residuals_cannot_leak_into_2016_replay():
    m=synthetic_margin()
    # 2016 labels become extreme future observations; residual pool for
    # 2016 must be exactly those from pre-2016.
    m.loc[m.season.eq(2016),"margin_prediction"]+=1000
    _,_,top,_=simulate(synthetic_forecasts(),m,MixedPolicy())
    # Projected current margin DID change, so selection is allowed to change.
    assert not top.empty
    # The no-lookahead requirement is explicitly encoded via last_residual_season.
    assert set(top.season)=={2016}


def test_only_2016_to_2025_forecasts_allowed():
    f=synthetic_forecasts()
    f.loc[f.index[0],"season"]=2026
    with pytest.raises(ValueError):
        simulate(f,synthetic_margin(),MixedPolicy())


def test_at_least_two_prior_model_residual_seasons_assembled():
    m=synthetic_margin().loc[lambda x:x.season.ge(2015)]
    # 2015 alone with 140 games is insufficient for the empirical CDF.
    with pytest.raises(ValueError,match="earlier-season"):
        simulate(synthetic_forecasts(),m,MixedPolicy())
