"""Exercise V7's full legacy-schedule -> live-quote -> ML/ATS pipeline."""
import numpy as np
import pandas as pd

from scripts.parlay_live_mixed import build_preview,parse_asof
from scripts.parlay_mixed_market_v5 import MixedPolicy


def test_v7_live_preview_end_to_end_and_masks_future_results():
    rng=np.random.default_rng(20261016)
    historical=[dict(game_id=f"historical_{i}",season=2024,
        week=i//10+1,gameday=str((pd.Timestamp("2024-01-01")+
            pd.Timedelta(days=i)).date()),gametime="13:00",
        game_type="REG",home_team="LA" if i%2 else "KC",
        away_team="ARI" if i%2 else "LAC",
        home_score=int(rng.integers(10,40)),away_score=int(rng.integers(10,40)),
        home_moneyline=-125,away_moneyline=115,spread_line=3.5)
        for i in range(240)]
    future=[dict(game_id=f"future_{i}",season=2026,week=6,
        gameday="2026-10-18",gametime="16:05",game_type="REG",
        home_team=h,away_team=a,home_score=100,away_score=0,
        home_moneyline=-250,away_moneyline=210,spread_line=7.5)
        for i,(h,a) in enumerate([("LA","ARI"),("KC","LAC")])]
    quotes=pd.DataFrame([dict(game_id=g["game_id"],bookmaker=b,
        captured_at="2026-10-16T19:30:00Z",home_moneyline=-250,
        away_moneyline=210,spread_line=7.5,home_spread_odds=-110,
        away_spread_odds=-110) for g in future for b in ("book_a","book_b")])
    past=pd.DataFrame(dict(season=[2024]*300,
        true_margin=rng.integers(-25,25,300),spread_line=[0.]*300,
        margin_prediction=[0.]*300))
    asof=parse_asof("2026-10-16T16:00:00-04:00")
    schedule=pd.DataFrame(historical+future)
    args=(schedule,quotes,past,asof,MixedPolicy())
    singles,parlays,weekly,report=build_preview(*args,strategy="v7")
    assert len(singles)==16
    assert set(singles.bookmaker)=={"book_a","book_b"}
    assert len(parlays)>0
    assert not singles.actual_result.notna().any()
    assert not parlays.research_unit_return.notna().any()
    assert report["games"]==2
    assert not report["forecast_calibration"]["2026"]["verified"]
    schedule.loc[schedule.game_id.str.startswith("future"),
        ["home_score","away_score"]]=[0,99]
    altered,changed,_,_=build_preview(*args,strategy="v7")
    pd.testing.assert_frame_equal(singles,altered)
    pd.testing.assert_frame_equal(parlays,changed)
