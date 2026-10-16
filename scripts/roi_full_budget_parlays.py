"""Full-stake mixed-market singles plus calculator-priced cross-game ML parlay research.

Singles: choose every qualifying game (no six-pick limit), spend exactly $100
per observed week; if no game qualifies, fall back to the highest-confidence
moneyline with a clearly labeled negative/unknown edge. Never promise EV.

Parlays: evaluate a distinct, optional sleeve using TWO rejected moneyline
legs in different games, no overlap with straight tickets. Joint probabilities
are illustrative independence products and parlay odds are standard compounded products
of historical recorded moneyline prices, NOT verified executable sportsbook quotes.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.roi_decision_v2 import candidates as model_candidates, allocate as v2_allocate
from scripts.roi_decision_v2 import report as v2_report, american_return

STRAIGHT_BUDGET = 100.
PARLAY_MAX_BUDGET = 15.
PARLAY_MIN_PROBABILITY = .60
PARLAY_MIN_JOINT_PROB = .36
PARLAY_MIN_ESTIMATED_EV = -.10


def compounded_parlay_decimal(*american_odds):
    """Calculate cross-game parlay decimal odds using Action Network's rule.

    Decimal prices multiply for independent legs from different games.
    https://www.actionnetwork.com/betting-calculators/parlay-calculator
    This calculation does not verify historical sportsbook availability.
    """
    if len(american_odds) < 2:
        raise ValueError("Parlays require at least two legs")
    return float(np.prod(1.0 + american_return(american_odds)))


def distribute_cents(weights, budget):
    """Exact allocation in integer cents with a deterministic tie break."""
    v = np.asarray(weights, dtype=float)
    if not len(v) or np.any(~np.isfinite(v)) or np.any(v < 0):
        raise ValueError("Stake weights must be finite, nonnegative, nonempty")
    if v.sum() <= 0:
        v = np.ones(len(v))
    target = int(round(100 * budget))
    real = v / v.sum() * target
    pennies = np.floor(real).astype(int)
    delta = target - int(pennies.sum())
    order = np.argsort(-(real - pennies), kind="stable")
    pennies[order[:delta]] += 1
    assert pennies.sum() == target
    return pennies / 100.


def make_singles(universe, budget=STRAIGHT_BUDGET):
    """All qualifying games, no bet-count cap, no more than one side per game.

    If there are no opportunities over the hurdle, the user explicitly wants
    forced expenditure. Use a high win-probability ML fallback, not a spurious
    high payout underdog edge. Label the forced bet as non-value research.
    """
    if not budget > 0:
        raise ValueError("budget must be positive")
    results = []
    for (_, _, _), slate in universe.groupby(["season", "week", "policy"], sort=True):
        candidates = slate.loc[slate.qualifies].sort_values(
            ["risk_score", "estimated_ev", "game_id", "kind", "side"],
            ascending=[False, False, True, True, True],
            kind="stable",
        ).drop_duplicates("game_id")
        fallback = candidates.empty
        if fallback:
            options = slate.loc[slate.kind.eq("ML")].sort_values(
                ["win_prob", "risk_score", "game_id", "side"],
                ascending=[False, False, True, True], kind="stable"
            )
            if options.empty:
                raise ValueError("No moneyline quote for fallback")
            candidates = options.head(1).copy()
        else:
            candidates = candidates.copy()
        candidates["forced_fallback"] = fallback
        candidates["selection_reason"] = (
            "FORCED_HIGH_PROBABILITY_ML_NO_VALUE"
            if fallback else "QUALIFYING_RISK_ADJUSTED_VALUE"
        )
        # Tilt to conservative probability and low payout; still retain score
        # ranking rather than giving every bet equal capital.
        weight = (
            np.maximum(.02, candidates.win_prob.to_numpy(float)) ** 2
            * np.exp(2.0 * np.clip(candidates.risk_score.to_numpy(float), -.25, .25))
            / np.sqrt(1 + candidates.payout.to_numpy(float))
        )
        candidates["stake"] = distribute_cents(weight, budget)
        candidates["profit"] = np.where(
            candidates.push, 0.,
            np.where(candidates.won, candidates.stake * candidates.payout, -candidates.stake),
        )
        assert candidates.game_id.nunique() == len(candidates)
        assert abs(candidates.stake.sum() - budget) < .005
        results.append(candidates)
    return pd.concat(results, ignore_index=True)


def parlay_from_rejected(universe, singles, max_budget=PARLAY_MAX_BUDGET):
    """One optional two-game ML parlay per week/policy from rejected candidates.

    Selection uses only pregame prices/probabilities. Never use actual outcomes
    for selection. Both legs must have different game_id and neither game may
    have an allocated straight ticket, to avoid double-counted exposure.
    """
    tickets = []
    reviewed = []
    keys = ["season", "week", "policy"]
    straight_games = {key: set(g.game_id) for key, g in singles.groupby(keys)}
    for key, slate in universe.groupby(keys, sort=True):
        rejected = slate.loc[
            slate.kind.eq("ML")
            & (~slate.qualifies)
            & slate.win_prob.ge(PARLAY_MIN_PROBABILITY)
            & (~slate.game_id.isin(straight_games.get(key, set())))
        ].sort_values(
            ["win_prob", "risk_score", "game_id", "side"],
            ascending=[False, False, True, True], kind="stable"
        ).drop_duplicates("game_id")
        if len(rejected) < 2:
            continue
        combinations = []
        for a, b in itertools.combinations(rejected.itertuples(index=False), 2):
            # Only different games; cross-game ML legs cannot push in NFL's
            # non-tied completed outcomes in the upstream dataset.
            joint_prob = float(a.win_prob * b.win_prob)
            calculated_decimal = compounded_parlay_decimal(a.odds, b.odds)
            ev = joint_prob * calculated_decimal - 1.
            if joint_prob < PARLAY_MIN_JOINT_PROB:
                continue
            candidate = {
                "season": key[0], "week": key[1], "policy": key[2],
                "game_a": a.game_id, "side_a": a.side,
                "game_b": b.game_id, "side_b": b.side,
                "leg_a_odds": a.odds, "leg_b_odds": b.odds,
                "leg_a_win_probability": a.win_prob,
                "leg_b_win_probability": b.win_prob,
                "joint_win_probability_independence": joint_prob,
                "calculated_parlay_decimal_odds": calculated_decimal,
                "estimated_ev_independence": ev,
                "both_won": bool(a.won and b.won),
                "independence_assumption": True,
                "historical_quote_executable": False,
            }
            combinations.append(candidate)
        if not combinations:
            continue
        ranked = pd.DataFrame(combinations).sort_values(
            ["estimated_ev_independence", "joint_win_probability_independence", "game_a", "game_b"],
            ascending=[False, False, True, True], kind="stable"
        )
        reviewed.extend(combinations)
        top = ranked.iloc[0].copy()
        if top.estimated_ev_independence < PARLAY_MIN_ESTIMATED_EV:
            continue
        # Independent RISK sleeve; not taken from $100 straights. Stake $2-$15
        # depending jointly on predicted hit rate and assessed EV.
        quality = max(0., min(1., (top.estimated_ev_independence + .10) / .20))
        confidence = max(0., min(1., (top.joint_win_probability_independence - .36) / .40))
        stake = round(2. + (max_budget - 2.) * quality * confidence, 2)
        top["stake"] = stake
        top["profit"] = stake * (
            float(top.calculated_parlay_decimal_odds - 1) if top.both_won else -1.
        )
        top["research_only"] = True
        tickets.append(top.to_dict())
    empty = pd.DataFrame()
    return (pd.DataFrame(tickets) if tickets else empty,
            pd.DataFrame(reviewed) if reviewed else empty)


def summarize(universe, singles, parlays, legacy):
    results = {"policies": {}, "parameters": {
        "weekly_straight_budget": STRAIGHT_BUDGET,
        "single_bet_cap": None, "fallback": "highest predicted win-probability ML",
        "parlay_max_extra_weekly": PARLAY_MAX_BUDGET,
        "parlay_legs": 2, "parlay_source": "rejected ML bets on other games",
        "parlay_price_type": "calculated from historical legs using standard Action Network decimal-odds multiplication, not a verified quoted sportsbook ticket",
    }, "warnings": [
        "Forced $100 weekly betting can place capital into a negative-EV market.",
        "More bets or parlays do not guarantee positive ROI; a negative-EV leg is not rescued by multiplication.",
        "Parlay odds are calculated with standard cross-game leg compounding, but historic leg prices do not establish simultaneous executable offers.",
        "Independence across games is assumed, and not necessarily true.",
        "Parlay stake is ADDITIONAL to the full $100 straight budget.",
        "Prior years have already been researched extensively; later years are not untouched validation.",
        "No live auto-betting or production promotion.",
    ]}
    legacy_report = v2_report(universe, legacy)
    for policy in sorted(universe.policy.unique()):
        table = []
        for label, y0, y1 in [
            ("development_2021_2022", 2021, 2022),
            ("later_2023_2025", 2023, 2025),
            ("all_2021_2025", 2021, 2025),
        ]:
            week_rows = universe.loc[
                universe.policy.eq(policy) & universe.season.between(y0, y1),
                ["season", "week"],
            ].drop_duplicates()
            s = singles[singles.policy.eq(policy) & singles.season.between(y0, y1)]
            p = (parlays[parlays.policy.eq(policy) & parlays.season.between(y0, y1)]
                 if not parlays.empty else pd.DataFrame())
            old = next(r for r in legacy_report["periods"][policy] if r["period"] == label)
            staked = float(s.stake.sum())
            profit = float(s.profit.sum())
            p_staked = float(p.stake.sum()) if len(p) else 0.
            p_profit = float(p.profit.sum()) if len(p) else 0.
            row = {
                "period": label, "weeks": len(week_rows),
                "single_bets": len(s), "single_ml": int(s.kind.eq("ML").sum()),
                "single_ats": int(s.kind.eq("ATS").sum()),
                "forced_fallback_bets": int(s.forced_fallback.sum()),
                "straight_staked": round(staked, 2),
                "straight_profit": round(profit, 2),
                "straight_roi": float(profit / staked) if staked else None,
                "parlay_tickets": len(p), "parlay_staked": round(p_staked, 2),
                "parlay_profit": round(p_profit, 2),
                "parlay_roi_calculated": float(p_profit / p_staked) if p_staked else None,
                "combined_staked": round(staked + p_staked, 2),
                "combined_profit": round(profit + p_profit, 2),
                "combined_roi_calculated": (float((profit + p_profit) / (staked + p_staked))
                                           if staked + p_staked else None),
                "prior_v2_bets": old["bets"], "prior_v2_staked": old["staked"],
                "prior_v2_profit": old["net"],
                "prior_v2_roi": old["roi_on_staked"],
            }
            if len(week_rows):
                expected_budget = STRAIGHT_BUDGET * len(week_rows)
                if abs(staked - expected_budget) > .01:
                    raise AssertionError("Must spend exactly $100 each week")
            table.append(row)
        results["policies"][policy] = table
    return results


def experiment(winner, margins):
    universe = model_candidates(winner, margins)
    singles = make_singles(universe)
    parlays, reviewed = parlay_from_rejected(universe, singles)
    legacy = v2_allocate(universe)
    comparison = summarize(universe, singles, parlays, legacy)
    return comparison, universe, singles, parlays, reviewed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--winner-predictions", required=True)
    p.add_argument("--margin-predictions", required=True)
    p.add_argument("--output", default="results/roi_full_budget_parlays")
    a = p.parse_args()
    comparison, universe, singles, parlays, reviewed = experiment(
        pd.read_csv(a.winner_predictions), pd.read_csv(a.margin_predictions)
    )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(comparison, indent=2))
    singles.to_csv(out / "straight_picks.csv", index=False)
    universe.to_csv(out / "market_checks.csv", index=False)
    if not parlays.empty:
        parlays.to_csv(out / "calculated_parlay_picks.csv", index=False)
    if not reviewed.empty:
        reviewed.to_csv(out / "parlay_candidates.csv", index=False)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
