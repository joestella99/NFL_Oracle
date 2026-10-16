"""Small edges, preference integrity, push math and executable preview flow."""
from dataclasses import replace
from math import prod

import numpy as np
import pandas as pd
import pytest

from nfl_oracle.conviction import (
    ConvictionPolicy, annotate_ticket, leg_key, liked_pick_audit, rank_views,
)
from scripts.parlay_conviction_v6 import decorate_legs, enumerate_slate, build_slates
from scripts.parlay_mixed_market_v5 import MixedPolicy, ticket, build_slates as build_v5


def leg(name, p=.8, ev=.01, q=0., kind="ML", won=True, pushed=False):
    dec = (1 + ev - q) / p
    return dict(game_id=name, season=2026, week=6, kind=kind, side="home",
                win_probability=p, push_probability=q, odds_decimal=dec,
                conservative_win_probability=max(0, p - .025),
                raw_leg_ev=ev, conservative_leg_ev=(p - .025) * dec * .99 + q - 1,
                candidate_selected=True, won=won, pushed=pushed)


def review(*legs, liked=frozenset()):
    return annotate_ticket(ticket(legs, MixedPolicy(), "football_base_prob"),
                           legs, ConvictionPolicy(), liked)


def test_one_percent_legs_visible_without_claiming_stress_pass():
    result = review(leg("LA"), leg("KC"))
    assert result["estimated_ev"] == pytest.approx(.0201)
    assert result["estimated_all_win_probability"] == pytest.approx(.64)
    assert result["qualifies_conviction"]
    assert not result["qualifies_value"]
    assert result["conviction_status"] == "POSITIVE_MODEL_EV_STRESS_FAILS"
    assert result["contains_low_edge_leg"]


def test_likely_winner_can_be_bad_price_and_cannot_be_rescued_by_like():
    bad = leg("LA", p=.95, ev=-.01)
    good = leg("KC", p=.8, ev=.1)
    result = review(bad, good, liked=frozenset({leg_key(bad)}))
    assert result["estimated_ev"] > 0
    assert not result["qualifies_conviction"]
    assert result["liked_leg_count"] == 1


def test_zero_edge_does_not_create_profit_and_marginal_math_explains_it():
    zero = review(leg("A", ev=0.), leg("B", ev=0.))
    assert not zero["qualifies_conviction"]
    a, b = leg("A", ev=.03), leg("B", ev=0.)
    result = review(a, b)
    assert result["estimated_ev"] == pytest.approx(.03)
    marginal_b = result["marginal_legs"][1]
    assert marginal_b["ticket_ev_change"] == pytest.approx(0)
    assert marginal_b["all_win_probability_change"] < 0
    assert marginal_b["ticket_break_even_leg_probability"] == pytest.approx(1 / (1.03 * b["odds_decimal"]))


def test_shorter_subsets_remain_available_and_same_game_never_doubled():
    legs = decorate_legs([leg("A"), leg("A", kind="ATS"), leg("B"), leg("C")], ConvictionPolicy())
    frame = enumerate_slate(legs, MixedPolicy(), ConvictionPolicy(), "football_base_prob")
    assert set(frame.legs) == {2, 3}
    assert all(len(set(row.split(" | "))) == n for row, n in zip(frame.game_ids, frame.legs))
    assert frame.loc[frame.conviction_rank.eq(1), "legs"].iloc[0] == 2


def test_confidence_rank_does_not_sort_on_biggest_ev_or_realized_outcome():
    strong = review(leg("A", .9, .01), leg("B", .9, .01))
    large_ev = review(leg("C", .67, .2), leg("D", .67, .2))
    frame = rank_views(pd.DataFrame([strong, large_ev]))
    assert frame.iloc[0].conviction_rank == 1
    assert frame.iloc[1].value_rank == 1
    mutated = frame.copy()
    mutated["true_unit_return"] = [-1, 100]
    again = rank_views(mutated)
    pd.testing.assert_series_equal(frame.conviction_rank, again.conviction_rank)
    assert frame.conviction_pareto.all()


def test_likes_change_only_preference_order_and_have_explicit_missing_audit():
    a, b, c = leg("A", .9), leg("B", .9), leg("C", .7)
    rows = decorate_legs([a, b, c], ConvictionPolicy())
    standard = enumerate_slate(rows, MixedPolicy(), ConvictionPolicy(), "football_base_prob")
    liked = enumerate_slate(rows, MixedPolicy(), ConvictionPolicy(), "football_base_prob",
                            frozenset({leg_key(c)}))
    for col in ["estimated_ev", "conservative_ev", "qualifies_conviction", "conviction_rank"]:
        pd.testing.assert_series_equal(standard[col], liked[col])
    assert "C:home:ML" in liked.loc[liked.preference_rank.eq(1), "selections"].iloc[0]
    legs = pd.DataFrame([dict(row, model="football_base_prob") for row in rows])
    audit = liked_pick_audit(legs, liked, frozenset({leg_key(c), "missing:home:ML"}))
    assert set(audit.reason) == {"HAS_CONVICTION_TICKET", "NO_MATCHING_QUOTED_PICK"}
    empty = liked_pick_audit(legs, pd.DataFrame(), frozenset({leg_key(c)}))
    assert empty.iloc[0].reason == "NO_PARTNER_TICKET_PASSES"


def test_push_expected_return_and_realized_void_are_preserved():
    a = leg("A", .7, .02, q=.1, kind="ATS", won=False, pushed=True)
    b = leg("B", .8, .01)
    result = review(a, b)
    assert result["estimated_ev"] == pytest.approx(1.02 * 1.01 - 1)
    assert result["true_unit_return"] == pytest.approx(b["odds_decimal"] - 1)
    assert result["estimated_all_win_probability"] == pytest.approx(.56)


def test_unplayed_ticket_has_no_invented_profit():
    result = review(leg("A", won=None, pushed=None), leg("B", won=None, pushed=None))
    assert result["true_unit_return"] is None


def test_v5_value_pool_is_unchanged_and_outcomes_do_not_affect_v6():
    from tests.test_parlay_mixed_market_v5 import game
    all_games = pd.DataFrame([game(i) for i in range(4)])
    margin_cols = ["game_id", "season", "week", "spread_line", "home_spread_odds",
                   "away_spread_odds", "margin_prediction", "true_margin"]
    margins = all_games[margin_cols]
    forecasts = all_games.drop(columns=margin_cols[3:])
    past = pd.DataFrame(dict(season=[2025] * 300, true_margin=[7.] * 300, margin_prediction=[8.] * 300))
    _, old, _ = build_v5(forecasts, margins, past, MixedPolicy(), variants=("market_home_prob",))
    _, new, _ = build_slates(forecasts, margins, past, MixedPolicy(), ConvictionPolicy(), variants=("market_home_prob",))
    assert set(old.loc[old.status.eq("QUALIFIES_RESEARCH"), "selections"]) == set(new.loc[new.qualifies_value, "selections"])
    changed = margins.copy()
    changed["true_margin"] = -100
    _, next_frame, _ = build_slates(forecasts, changed, past, MixedPolicy(), ConvictionPolicy(), variants=("market_home_prob",))
    for col in ["selections", "estimated_ev", "qualifies_conviction", "conviction_rank", "value_rank"]:
        pd.testing.assert_series_equal(new[col], next_frame[col])


@pytest.mark.parametrize("changes", [{"max_legs": True}, {"max_legs": 6}, {"min_leg_ev": -.01}, {"min_ticket_ev": float("nan")}])
def test_invalid_policy(changes):
    with pytest.raises(ValueError):
        replace(ConvictionPolicy(), **changes).validate()


def test_live_preview_runs_full_model_to_ticket_pipeline_and_keeps_books_separate():
    from scripts.parlay_live_mixed import build_preview, parse_asof
    rng = np.random.default_rng(16)
    history = [dict(game_id=f"old{i}", season=2024, week=i // 10 + 1,
        gameday=str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date()),
        gametime="13:00", game_type="REG", home_team="LA" if i % 2 else "KC",
        away_team="ARI" if i % 2 else "LAC", home_score=int(rng.integers(10, 40)),
        away_score=int(rng.integers(10, 40)), home_moneyline=-125, away_moneyline=115,
        spread_line=2.5) for i in range(240)]
    future = [dict(game_id=f"future{i}", season=2026, week=6, gameday="2026-10-18",
        gametime="16:05", game_type="REG", home_team=h, away_team=a,
        home_score=100, away_score=0, home_moneyline=-250, away_moneyline=210,
        spread_line=7.5) for i, (h, a) in enumerate([("LA", "ARI"), ("KC", "LAC")])]
    quotes = pd.DataFrame([dict(game_id=g["game_id"], bookmaker=book,
        captured_at="2026-10-16T19:30:00Z", home_moneyline=-250,
        away_moneyline=210, spread_line=7.5, home_spread_odds=-110,
        away_spread_odds=-110) for g in future for book in ["one", "two"]])
    past = pd.DataFrame(dict(season=[2024] * 300,
        true_margin=rng.integers(-25, 25, 300), margin_prediction=[0.] * 300))
    schedule = pd.DataFrame(history + future)
    args = (schedule, quotes, past, parse_asof("2026-10-16T16:00:00-04:00"), MixedPolicy())
    legs, tickets, weeks, report = build_preview(*args, conviction_policy=ConvictionPolicy())
    assert not tickets.empty
    assert set(tickets.bookmaker) == {"one", "two"}
    assert tickets.true_unit_return.isna().all()
    assert report["matched_games"] == 2
    assert not legs.won.notna().any()
    schedule.loc[schedule.game_id.str.startswith("future"), ["home_score", "away_score"]] = [0, 99]
    _, changed, _, _ = build_preview(*args, conviction_policy=ConvictionPolicy())
    for col in ["selections", "estimated_ev", "qualifies_conviction", "conviction_rank"]:
        pd.testing.assert_series_equal(tickets[col], changed[col])
