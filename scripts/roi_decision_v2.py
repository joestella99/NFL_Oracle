"""Risk-aware, cross-market NFL Oracle decision engine (research, not live betting).

All calibrations use strictly earlier seasons. Historical nflverse sportsbook
quotes have no verified executable timestamp and cannot establish real profits.
A strong favorite triggers ML price review, NOT an automatic moneyline bet.
A model prediction 0.9 points away from the spread triggers uncertainty
evaluation, never automatic rejection. No same-game "hedging" assumption.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.odds_aware_alternatives import prepare
from scripts.consensus_allocator import returns

FOOTBALL_SIGNALS = [
    "epa_matchup_diff", "qb_pass_epa_diff", "run_attack_vs_defense",
    "epa_trimmed_diff", "success_rate_diff", "epa_std_diff",
]
# Experimental policy settings, not optimized against historical ROI.
POLICIES = {
    "balanced": dict(ml_blend=.35, spread_blend=.30, min_ev=.005, uncertainty_cost=.35),
    "risk_tolerant": dict(ml_blend=.60, spread_blend=.45, min_ev=-.01, uncertainty_cost=.15),
}
MAX_WEEKLY_RISK = 100.
MAX_BETS = 6


def american_return(odds):
    arr = np.asarray(odds, dtype=float)
    if not np.isfinite(arr).all() or ((arr > -100) & (arr < 100)).any():
        raise ValueError("Missing or invalid American odds")
    return returns(arr)


def market_probs(home, away):
    ih = 1 / (1 + american_return([home])[0])
    ia = 1 / (1 + american_return([away])[0])
    return float(ih / (ih + ia)), float(ia / (ih + ia))


def empirical_cover(margin_forecast, line, earlier_residuals):
    """Discrete score margins: exact spread pushes refund rather than lose.

    This is a retrospective empirical *residual* model, not a claim that
    historical outcome residuals represent perfectly calibrated distributions.
    """
    residuals = np.asarray(earlier_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 200:
        raise ValueError("Need >=200 earlier-season held-out margin errors")
    simulated = np.rint(float(margin_forecast) + residuals)
    relative = simulated - float(line)
    return (
        float(np.mean(relative > 0)),
        float(np.mean(relative < 0)),
        float(np.mean(np.isclose(relative, 0))),
    )


def candidates(winner_predictions, margin_predictions, first=2021):
    calibr = prepare(winner_predictions)
    primary = winner_predictions.loc[
        winner_predictions.model.eq("baseline"),
        ["game_id", "market_home_prob"] + FOOTBALL_SIGNALS,
    ]
    if primary.game_id.duplicated().any():
        raise ValueError("Duplicated pregame statistics")
    forecasts = (
        margin_predictions.merge(
            calibr[["game_id", "calibrated", "home_moneyline", "away_moneyline"]],
            on="game_id", validate="one_to_one",
        ).merge(primary, on="game_id", validate="one_to_one")
    )
    out = []
    for year in sorted(forecasts.season.unique()):
        if year < first:
            continue
        historic = margin_predictions.loc[margin_predictions.season < year]
        if historic.season.nunique() < 2:
            continue
        residuals = (historic.true_margin - historic.margin_prediction).to_numpy(float)
        rows = forecasts.loc[forecasts.season.eq(year)]
        for row in rows.itertuples(index=False):
            if not np.isfinite(row.spread_line):
                continue
            home_ml_market, away_ml_market = market_probs(row.home_moneyline, row.away_moneyline)
            home_ats_market, away_ats_market = market_probs(row.home_spread_odds, row.away_spread_odds)
            home_cover, away_cover, push = empirical_cover(
                row.margin_prediction, row.spread_line, residuals
            )
            # Market ATS probability is CONDITIONAL on not pushing. Distribute
            # the empirically estimated push mass before blending.
            cover_market_home = (1 - push) * home_ats_market
            cover_market_away = (1 - push) * away_ats_market
            p_home = float(row.calibrated)
            strong_favorite = max(p_home, 1 - p_home) >= .80
            fside = "home" if p_home >= .5 else "away"
            gap = float(row.margin_prediction - row.spread_line)
            adverse_signals = {}
            for signal in FOOTBALL_SIGNALS:
                val = getattr(row, signal)
                adverse_signals[signal] = float(val) if np.isfinite(val) else np.nan
            for policy_name, policy in POLICIES.items():
                ml_p_home = (1 - policy["ml_blend"]) * home_ml_market + policy["ml_blend"] * p_home
                spread_p_home = (1 - policy["spread_blend"]) * cover_market_home + policy["spread_blend"] * home_cover
                spread_p_away = (1 - policy["spread_blend"]) * cover_market_away + policy["spread_blend"] * away_cover
                for kind, side, odds, chance, baseline in [
                    ("ML", "home", row.home_moneyline, ml_p_home, home_ml_market),
                    ("ML", "away", row.away_moneyline, 1 - ml_p_home, away_ml_market),
                    ("ATS", "home", row.home_spread_odds, spread_p_home, cover_market_home),
                    ("ATS", "away", row.away_spread_odds, spread_p_away, cover_market_away),
                ]:
                    refund = push if kind == "ATS" else 0.
                    # EV is Win * net return - Lose. A push returns the stake.
                    win = float(chance)
                    lose = max(0., 1. - win - refund)
                    payout = float(american_return([odds])[0])
                    ev = win * payout - lose
                    # Rough uncertainty proxy: more model-market divergence
                    # means less certainty. Tested as a penalty, NOT a bound.
                    uncertainty = (
                        (.03 if kind == "ML" else .035)
                        + .3 * abs(win - baseline)
                    )
                    penalty = policy["uncertainty_cost"] * uncertainty * (1 + payout)
                    score = ev - penalty
                    opposing = strong_favorite and side != fside
                    required = policy["min_ev"] + (.015 if opposing else 0)
                    qualifies = score >= required
                    actual_margin = float(row.true_margin)
                    if kind == "ML":
                        won = actual_margin > 0 if side == "home" else actual_margin < 0
                        pushed = False
                    else:
                        actual_cover = actual_margin - float(row.spread_line)
                        won = actual_cover > 0 if side == "home" else actual_cover < 0
                        pushed = bool(np.isclose(actual_cover, 0))
                    out.append({
                        "game_id": row.game_id, "season": year, "week": row.week,
                        "policy": policy_name, "kind": kind, "side": side,
                        "odds": float(odds), "win_prob": win, "push_prob": refund,
                        "market_win_prob": float(baseline), "estimated_ev": ev,
                        "risk_score": score, "qualifies": bool(qualifies),
                        "strong_favorite_review": bool(strong_favorite),
                        "favorite_side": fside, "against_strong_favorite": bool(opposing),
                        "predicted_margin": float(row.margin_prediction),
                        "market_home_margin": float(row.spread_line),
                        "margin_gap": gap, "observed_margin": actual_margin,
                        "won": bool(won), "push": pushed, "payout": payout,
                        "selected_team_lost_but_covered": bool(
                            kind == "ATS" and won and
                            ((side == "home" and actual_margin < 0) or
                             (side == "away" and actual_margin > 0))
                        ),
                        **adverse_signals,
                    })
    return pd.DataFrame(out)


def allocate(candidates_df, risk_budget=MAX_WEEKLY_RISK, max_bets=MAX_BETS):
    bets = []
    for (_, _, policy), week in candidates_df.groupby(["season", "week", "policy"]):
        if not week.qualifies.any():
            continue
        choice = (
            week.loc[week.qualifies]
            .sort_values(["risk_score", "estimated_ev", "game_id", "kind", "side"],
                         ascending=[False, False, True, True, True])
            .drop_duplicates("game_id")
            .head(max_bets)
            .copy()
        )
        if choice.empty:
            continue
        # Fixed exposure per qualified wager. No obligation to bet the $100.
        # We cannot claim comparable returns when fewer dollars are at risk:
        # report total ROI and profit as % of the available weekly budget.
        choice["stake"] = round(risk_budget / max_bets, 2)
        choice["profit"] = np.where(choice.push, 0.,
            np.where(choice.won, choice.stake * choice.payout, -choice.stake))
        bets.append(choice)
    return pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()


def report(universe, bets):
    result = {
        "policies": POLICIES,
        "risk_budget_per_week": MAX_WEEKLY_RISK,
        "max_bets_per_week": MAX_BETS,
        "periods": {},
        "limitations": [
            "Historical odds/spreads have no verified timestamp or guaranteed execution.",
            "2019-25 have been examined repeatedly; later years are not an untouched holdout.",
            "Spread win distributions are inferred from earlier-season model errors; "
            "tail calibration and game-to-game dependence remain uncertain.",
            "An available wager is not automatically a favorable wager.",
            "ML+same-team favorite ATS is NOT a hedge or independent same-game parlay.",
            "No parlay payout is evaluated without a verified quote.",
        ],
    }
    for policy in POLICIES:
        all_games = universe.loc[universe.policy.eq(policy)]
        selection = bets.loc[bets.policy.eq(policy)] if len(bets) else pd.DataFrame()
        r = []
        for label, first, last in [
            ("development_2021_2022", 2021, 2022),
            ("later_2023_2025", 2023, 2025),
            ("all_2021_2025", 2021, 2025),
        ]:
            period = all_games[all_games.season.between(first, last)]
            b = selection[selection.season.between(first, last)] if len(selection) else pd.DataFrame()
            weeks = period[["season", "week"]].drop_duplicates().shape[0]
            spent = float(b.stake.sum()) if len(b) else 0.
            profit = float(b.profit.sum()) if len(b) else 0.
            r.append({
                "period": label, "weeks": int(weeks), "bets": int(len(b)),
                "moneylines": int((b.kind == "ML").sum()) if len(b) else 0,
                "spreads": int((b.kind == "ATS").sum()) if len(b) else 0,
                "risk_budget_available": weeks * MAX_WEEKLY_RISK,
                "staked": round(spent, 2), "net": round(profit, 2),
                "roi_on_staked": float(profit / spent) if spent else None,
                "return_on_available_budget": float(profit / (weeks * MAX_WEEKLY_RISK)) if weeks else None,
                "winning_weeks": int((b.groupby(["season", "week"]).profit.sum() > 0).sum()) if len(b) else 0,
                "losing_team_ats_covers": int(b.selected_team_lost_but_covered.sum()) if len(b) else 0,
            })
        result["periods"][policy] = r
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--winner-predictions", required=True)
    p.add_argument("--margin-predictions", required=True)
    p.add_argument("--output", default="results/roi_decision_v2")
    args = p.parse_args()
    winner = pd.read_csv(args.winner_predictions)
    margin = pd.read_csv(args.margin_predictions)
    universe = candidates(winner, margin)
    bets = allocate(universe)
    results = report(universe, bets)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "report.json").write_text(json.dumps(results, indent=2))
    universe.to_csv(dst / "all_market_checks.csv", index=False)
    bets.to_csv(dst / "bets.csv", index=False)
    if len(bets):
        bets.loc[~bets.won & ~bets.push].to_csv(dst / "missed_bets.csv", index=False)
        segment = bets.assign(
            favorite_opponent=bets.against_strong_favorite,
            margin_gap_bucket=pd.cut(abs(bets.margin_gap),
                [-.01, 1, 3, 7, np.inf], labels=["0-1", "1-3", "3-7", "7+"]),
        ).groupby(["policy", "kind", "margin_gap_bucket"], observed=True).agg(
            count=("game_id", "size"), staked=("stake", "sum"),
            profit=("profit", "sum")).reset_index()
        segment["roi"] = segment.profit / segment.staked
        segment.to_csv(dst / "miss_trends.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
