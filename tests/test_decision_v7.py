"""V7 decision engine tests: no hindsight, complete market audits, small edges."""
import numpy as np
import pandas as pd
import pytest

from nfl_oracle.decision_v7 import (
    V7Policy, Calibration, fit_calibration, game_markets,
    evaluate_slate, _ticket, _candidate_options,
)


def cal():
    # Pretend this is a *prior-season* OOS residual sample.
    return Calibration(1.0, 1.0, tuple(range(-15,16))*10, 310, 2025, True)


def quotes():
    return pd.DataFrame([
        dict(game_id="rams",season=2026,week=6,bookmaker="A",
             home_team="LA",away_team="ARI",home_moneyline=-850,
             away_moneyline=600,home_spread_odds=-110,
             away_spread_odds=-110,spread_line=12.5,
             model_home_prob=.90,margin_prediction=12.5,true_margin=np.nan,
             captured_at="2026-10-16T19:30:00Z",kickoff_utc="2026-10-18T20:05:00Z"),
        dict(game_id="chiefs",season=2026,week=6,bookmaker="A",
             home_team="KC",away_team="LAC",home_moneyline=-390,
             away_moneyline=310,home_spread_odds=-110,
             away_spread_odds=-110,spread_line=7.5,
             model_home_prob=.80,margin_prediction=7.5,true_margin=np.nan,
             captured_at="2026-10-16T19:30:00Z",kickoff_utc="2026-10-18T20:25:00Z"),
        dict(game_id="other",season=2026,week=6,bookmaker="A",
             home_team="PHI",away_team="DAL",home_moneyline=-190,
             away_moneyline=160,home_spread_odds=-110,
             away_spread_odds=-110,spread_line=3.5,
             model_home_prob=.75,margin_prediction=5,true_margin=np.nan,
             captured_at="2026-10-16T19:30:00Z",kickoff_utc="2026-10-18T20:25:00Z"),
    ])


def run(data=None,likes=frozenset()):
    return evaluate_slate(quotes() if data is None else data,cal(),liked=likes,
        live_asof=pd.Timestamp("2026-10-16T20:00:00Z"))


def test_every_ml_and_spread_side_audited_even_if_rejected():
    straight,parlays,_=run()
    assert len(straight)==12
    assert set(straight.kind)=={"ML","ATS"}
    assert straight.groupby(["game_id","kind"]).side.nunique().eq(2).all()
    assert len(parlays)>0
    assert all(len(s.split(" | "))==len(set(s.split(" | "))) for s in parlays.game_ids)
    assert set(parlays.bookmaker)=={"A"}
    assert set(parlays.leg_count).issubset({2,3})


def test_low_ev_high_confidence_favorites_survive_in_parlays():
    singles,parlays,_=run()
    r=singles.set_index("selection").loc["rams:home:ML"]
    c=singles.set_index("selection").loc["chiefs:home:ML"]
    assert 0 < r.raw_ev < .02
    assert 0 < c.raw_ev < .02
    joined=parlays.loc[parlays.selections.str.contains("rams:home:ML")
                      & parlays.selections.str.contains("chiefs:home:ML")]
    assert not joined.empty
    assert joined.raw_ev.gt(0).any()


def test_confidence_without_price_value_does_not_claim_ev():
    single=game_markets(dict(quotes().iloc[0],model_home_prob=.86),
                         cal(),V7Policy())
    bad=next(x for x in single if x["kind"]=="ML" and x["side"]=="home")
    assert bad["win_probability"]>=.70
    assert bad["raw_ev"]<0
    assert bad["status"]=="HIGH_CONFIDENCE_BAD_PRICE"


def test_preferred_likes_never_change_selection_probabilities():
    first,initial,_=run()
    liked=frozenset({"rams:home:ML"})
    second,preferred,audit=run(likes=liked)
    pd.testing.assert_frame_equal(first.drop(columns=["liked","liked_rank"]),
                                  second.drop(columns=["liked","liked_rank"]))
    cols=["selections","raw_ev","stressed_ev","full_payout_win_probability"]
    pd.testing.assert_frame_equal(initial[cols],preferred[cols])
    assert not audit.empty
    assert audit.iloc[0].selection=="rams:home:ML"
    assert preferred.loc[preferred.selections.str.contains("rams:home:ML"),"liked_count"].gt(0).all()


def test_future_final_scores_never_change_any_selection_rank():
    a=quotes()
    x,p,_=run(a)
    a["true_margin"]=[-50,80,9]
    y,q,_=run(a)
    pd.testing.assert_frame_equal(x,y)
    pd.testing.assert_frame_equal(p,q)


def test_no_cross_book_parlays_or_cross_kickoff_prices():
    a=quotes()
    duplicate=a.copy()
    duplicate["bookmaker"]="B"
    data=pd.concat([a,duplicate])
    s,p,_=run(data)
    assert len(s)==24
    assert set(p.bookmaker)=={"A","B"}
    a.loc[0,"captured_at"]="2026-10-18T22:00:00Z"
    s,p,_=run(a)
    assert not s.game_id.eq("rams").any()
    assert not p.game_ids.str.contains("rams").any()


def test_quote_skew_excludes_unrelated_snapshot():
    a=quotes()
    a.loc[0,"captured_at"]="2026-10-16T17:30:00Z"
    _,p,_=run(a)
    assert not p.game_ids.str.contains("rams").any()


def test_refund_on_spread_push_and_full_loss_if_any_other_leg_fails():
    policy=V7Policy()
    x={"bookmaker":"A","season":2026,"week":6,"game_id":"a",
       "kind":"ATS","side":"home","selection":"a:home:ATS",
       "win_probability":.45,"stressed_win_probability":.43,
       "push_probability":.10,"decimal_odds":1.9,"raw_ev":-.045,
       "actual_result":"push"}
    y=dict(x,game_id="b",selection="b:home:ML",kind="ML",
           push_probability=0.,actual_result="win",decimal_odds=1.6)
    t=_ticket((x,y),policy,frozenset())
    assert t["research_unit_return"]==pytest.approx(.6)
    assert t["raw_ev"]==pytest.approx((.45*1.9+.10)*(.45*1.6)-1)
    y["actual_result"]="loss"
    assert _ticket((x,y),policy,frozenset())["research_unit_return"]==-1.


def test_calibration_uses_previous_seasons_and_prefers_better_market():
    rng=np.random.default_rng(47)
    n=480
    seasons=np.repeat([2023,2024,2025,2026],120)
    markets=rng.uniform(.28,.72,n)
    outcomes=rng.binomial(1,markets)
    margins=np.where(outcomes,7.,-7.)+rng.normal(0,4,n)
    df=pd.DataFrame(dict(game_id=[str(i) for i in range(n)],season=seasons,
        home_win=outcomes,market_home_prob=markets,prediction=1-markets,
        spread_line=np.where(markets>.5,3.,-3.),
        margin_prediction=rng.normal(0,10,n),true_margin=margins))
    before=fit_calibration(df,2026)
    assert before.calibrated
    assert before.last_training_season==2025
    assert before.moneyline_weight==0
    df.loc[df.season.eq(2026),"home_win"]=1-df.loc[df.season.eq(2026),"home_win"]
    df.loc[df.season.eq(2026),"true_margin"]=-999
    after=fit_calibration(df,2026)
    assert before==after


def test_missing_margin_history_is_explicit_not_fabricated():
    s,p,_=evaluate_slate(quotes(),Calibration.market_fallback(),live_asof=
        pd.Timestamp("2026-10-16T20:00:00Z"))
    ats=s.loc[s.kind.eq("ATS")]
    assert ats.status.eq("MISSING_PRIOR_MARGIN_CALIBRATION").all()
    assert not s.loc[s.kind.eq("ML")].raw_ev.gt(0).any()
    assert p.empty or p.raw_ev.le(0).all()


def test_reject_stale_missing_or_duplicated_quotes():
    with pytest.raises(ValueError):
        evaluate_slate(quotes(),cal(),live_asof="2026-10-16T16:00:00")
    with pytest.raises(ValueError):
        run(pd.concat([quotes(),quotes().head(1)],ignore_index=True))


def test_admissible_liked_alternative_is_not_lost_to_shortlist_truncation():
    legs=[
        dict(game_id="x",kind="ML",side="home",selection="x:home:ML",
             candidate=True,win_probability=.85,raw_ev=.04),
        dict(game_id="x",kind="ML",side="away",selection="x:away:ML",
             candidate=True,win_probability=.15,raw_ev=.005),
        dict(game_id="x",kind="ATS",side="home",selection="x:home:ATS",
             candidate=True,win_probability=.53,raw_ev=.03),
        dict(game_id="x",kind="ATS",side="away",selection="x:away:ATS",
             candidate=True,win_probability=.47,raw_ev=.005),
    ]
    baseline=_candidate_options(legs,V7Policy())
    assert "x:away:ML" not in {r["selection"] for r in baseline}
    liked=_candidate_options(legs,V7Policy(),frozenset({"x:away:ML"}))
    assert "x:away:ML" in {r["selection"] for r in liked}
    assert any(r["raw_ev"]==.005 for r in liked)
