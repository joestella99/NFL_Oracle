"""Percentage-only ticket review: separate confidence from price robustness.

Model-positive, small-edge tickets belong in a visible research view even
when a probability stress test erases the edge. They are never labelled
robust on that basis. Ranking uses no outcomes and does not allocate stakes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, prod

import pandas as pd


@dataclass(frozen=True)
class ConvictionPolicy:
    min_leg_win_probability: float = .65
    min_leg_ev: float = 0.
    min_ticket_win_probability: float = .40
    min_ticket_ev: float = .005
    max_legs: int = 3

    def validate(self):
        for key, value in asdict(self).items():
            if key == "max_legs":
                if type(value) is not int or not 2 <= value <= 4:
                    raise ValueError("max_legs must be an integer from 2 to 4")
            elif not isfinite(value) or not 0 <= value < 1:
                raise ValueError("Invalid conviction threshold: " + key)


def leg_key(leg: dict) -> str:
    return f'{leg["game_id"]}:{leg["side"]}:{leg["kind"]}'


def conviction_leg(leg: dict, policy: ConvictionPolicy) -> bool:
    return (leg["win_probability"] >= policy.min_leg_win_probability
            and leg["raw_leg_ev"] >= policy.min_leg_ev)


def annotate_ticket(t: dict, legs: tuple[dict, ...], policy: ConvictionPolicy,
                    liked: frozenset[str] = frozenset()) -> dict:
    """Audit raw/stressed views and the marginal effect of each leg.

Independent games and multiplied decimal prices are assumptions inherited
from V5. With pushes, gross expectation is product(p_win * decimal + p_push).
Adding a leg multiplies expected gross by its own factor; it always reduces
the probability of receiving the full advertised payout.
"""
    factors = [x["win_probability"] * x["odds_decimal"]
               + x["push_probability"] for x in legs]
    gross = prod(factors)
    eligible = (len(legs) <= policy.max_legs
                and all(conviction_leg(x, policy) for x in legs)
                and t["estimated_all_win_probability"] >= policy.min_ticket_win_probability
                and t["estimated_ev"] >= policy.min_ticket_ev)
    value = t["status"] == "QUALIFIES_RESEARCH"
    members = [leg_key(x) for x in legs]
    marginal = []
    for i, leg in enumerate(legs):
        rest = legs[:i] + legs[i + 1:]
        rest_gross = prod(factors[:i] + factors[i + 1:])
        # Required win probability with all other leg prices/probabilities
        # held fixed, retaining this leg's predicted void/refund mass.
        required = (1 / rest_gross - leg["push_probability"]) / leg["odds_decimal"]
        marginal.append(dict(
            selection=members[i], raw_leg_ev=leg["raw_leg_ev"],
            ticket_ev_without_leg=rest_gross - 1,
            ticket_ev_change=gross - rest_gross,
            all_win_probability_change=(t["estimated_all_win_probability"]
                                        - prod(x["win_probability"] for x in rest)),
            ticket_break_even_leg_probability=required,
            ticket_probability_cushion=leg["win_probability"] - required,
        ))
    status = ("ROBUST_UNDER_V5_STRESS" if eligible and value else
              "POSITIVE_MODEL_EV_STRESS_FAILS" if eligible else
              "NOT_IN_CONVICTION_VIEW")
    return dict(t, qualifies_value=value, qualifies_conviction=eligible,
                conviction_status=status,
                liked_leg_count=sum(x in liked for x in members),
                liked_selections=" | ".join(x for x in members if x in liked),
                marginal_legs=marginal,
                # ROI percentage on a hit, distinct from expected return.
                full_payout_profit=t["price_proxy_decimal"] - 1,
                contains_low_edge_leg=any(0 <= x["raw_leg_ev"] < .02 for x in legs))


def rank_views(tickets: pd.DataFrame) -> pd.DataFrame:
    """Confidence-first and EV-first ranks coexist; no combined arbitrary score.

Pareto-efficient conviction tickets have no other ticket with both a higher
full-payout hit probability and at least as much EV (or vice versa). User
likes change the preference rank only, never probability or qualification.
"""
    if tickets.empty:
        return tickets.copy()
    result = tickets.copy()
    for name in ("conviction_rank", "value_rank", "preference_rank"):
        result[name] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["conviction_pareto"] = False
    group = [c for c in ("season", "week", "model", "bookmaker") if c in result]
    for _, frame in result.groupby(group, dropna=False, sort=False):
        conviction = frame.loc[frame.qualifies_conviction].sort_values(
            ["estimated_all_win_probability", "estimated_ev", "legs", "selections"],
            ascending=[False, False, True, True], kind="stable")
        value = frame.loc[frame.qualifies_value].sort_values(
            ["conservative_ev", "estimated_all_win_probability", "selections"],
            ascending=[False, False, True], kind="stable")
        preferred = conviction.sort_values(
            ["liked_leg_count", "estimated_all_win_probability", "estimated_ev", "selections"],
            ascending=[False, False, False, True], kind="stable")
        for name, ordered in (("conviction_rank", conviction), ("value_rank", value),
                              ("preference_rank", preferred)):
            result.loc[ordered.index, name] = list(range(1, len(ordered) + 1))
        best_prior_ev = float("-inf")
        # Process probability ties together so identical points all survive.
        for _, tied in conviction.groupby("estimated_all_win_probability", sort=False):
            best_tied = tied.estimated_ev.max()
            ids = tied.index[(tied.estimated_ev == best_tied) & (tied.estimated_ev > best_prior_ev)]
            result.loc[ids, "conviction_pareto"] = True
            best_prior_ev = max(best_prior_ev, best_tied)
    return result


def liked_pick_audit(legs: pd.DataFrame, tickets: pd.DataFrame,
                     liked: frozenset[str]) -> pd.DataFrame:
    """Every explicitly liked pick gets a visible result, including exclusions."""
    rows = []
    for key in sorted(liked):
        found = legs.loc[legs.selection.eq(key)] if len(legs) else pd.DataFrame()
        if found.empty:
            rows.append(dict(selection=key, reason="NO_MATCHING_QUOTED_PICK"))
            continue
        groups = [c for c in ("season", "week", "model", "bookmaker") if c in found]
        for _, sub in found.groupby(groups, dropna=False):
            leg = sub.iloc[0]
            eligible = tickets
            if len(eligible):
                for col in groups:
                    eligible = eligible.loc[eligible[col].eq(leg[col])]
            eligible = eligible.loc[eligible.qualifies_conviction & eligible.selections.map(
                lambda text: key in text.split(" | "))] if len(eligible) else eligible
            best = eligible.sort_values("conviction_rank").head(1) if len(eligible) else eligible
            rows.append(dict(
                **{c: leg[c] for c in groups}, selection=key,
                win_probability=leg.win_probability, raw_leg_ev=leg.raw_leg_ev,
                reason=("HAS_CONVICTION_TICKET" if len(best) else
                        "LEG_BELOW_CONVICTION_STANDARD" if not leg.conviction_candidate else
                        "NO_PARTNER_TICKET_PASSES"),
                best_ticket=best.iloc[0].selections if len(best) else None,
            ))
    return pd.DataFrame(rows)
