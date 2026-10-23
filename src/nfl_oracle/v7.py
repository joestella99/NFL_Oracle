"""V7 guarded decision layer. Does not alter archived V5/V6 research.

Input: point-in-time market-side audits from the existing forecast pipeline.
This module never treats sportsbook favorites or a model label as evidence of edge.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
from math import isfinite, prod
import pandas as pd


@dataclass(frozen=True)
class V7Policy:
    min_calibration_games: int = 200
    min_probability_bucket_games: int = 35
    min_single_ev: float = .015
    min_ticket_ev: float = .03
    min_ticket_probability: float = .15
    probability_stress: float = .025
    price_stress: float = .01
    max_legs: int = 3
    max_sides_per_game: int = 2


def _valid_probability(x):
    try:
        n = float(x)
        return n if isfinite(n) and 0 <= n <= 1 else None
    except (TypeError, ValueError):
        return None


def calibrate_legs(audit: pd.DataFrame, history: pd.DataFrame, policy=V7Policy(),
                   asof_season=None, asof_week=None):
    """Past-only reliability correction by model, market kind and probability bucket.

    Uses *prior games only*, and shrinks the predicted probability toward the
    empirical bucket rate. Requires a strict chronological reference set;
    missing or inadequate history results in ABSTAIN, not zero uncertainty.
    A history row needs model, kind, win_probability, won, season, week.
    """
    required = {"model","kind","win_probability","won","season","week"}
    if not required.issubset(history):
        raise ValueError("Historical calibration needs " + ", ".join(sorted(required)))
    if asof_season is None or asof_week is None:
        raise ValueError("Explicit season/week cutoff required")
    prior = history.loc[(history.season < asof_season) |
                        ((history.season == asof_season) & (history.week < asof_week))].copy()
    prior = prior.loc[prior.won.notna()].copy()
    prior["win_probability"] = pd.to_numeric(prior.win_probability, errors="coerce")
    prior = prior.loc[prior.win_probability.between(0,1)]
    prior["bucket"] = (prior.win_probability * 10).clip(upper=9.999).astype(int)
    out = audit.copy()
    records = []
    for r in out.to_dict("records"):
        p = _valid_probability(r.get("win_probability"))
        group = prior.loc[(prior.model == r.get("model")) & (prior.kind == r.get("kind"))]
        bucket = -1 if p is None else min(int(p*10),9)
        near = group.loc[group.bucket.eq(bucket)]
        count, local = len(group), len(near)
        record = dict(r, calibration_samples=count, bucket_samples=local,
                      estimated_calibrated_probability=None, lower_probability=None,
                      conservative_ev=None, v7_leg_status="ABSTAIN_NO_CALIBRATION")
        if p is None:
            record["v7_leg_status"] = "INVALID_PROBABILITY"
        elif count >= policy.min_calibration_games and local >= policy.min_probability_bucket_games:
            # Shrink toward an empirical *past-only* reliability estimate.
            # This is a policy stress, NOT a statistical confidence interval.
            empirical = float(near.won.astype(float).mean())
            calibrated = .5*p + .5*empirical
            low = max(0., calibrated-policy.probability_stress)
            decimal = float(r.get("odds_decimal",float("nan")))
            push = float(r.get("push_probability",0))
            if isfinite(decimal) and decimal > 1 and 0 <= push <= 1:
                ev = low*decimal*(1-policy.price_stress)+push-1
                record.update(estimated_calibrated_probability=calibrated,
                              lower_probability=low, conservative_ev=ev,
                              v7_leg_status=("CANDIDATE" if ev>=policy.min_single_ev
                                             else "NO_VERIFIED_EDGE"))
            else:
                record["v7_leg_status"] = "INVALID_PRICE"
        records.append(record)
    return pd.DataFrame(records)


def tickets_for_slate(legs: pd.DataFrame, policy=V7Policy()):
    """Audit both moneyline sides and both spread sides; never same-game legs.

    Each ticket contains independent *games* as an explicit approximation,
    and is labelled research-only pending independently accepted bookmaker quote.
    """
    if legs.empty:
        return pd.DataFrame()
    qualifying = legs.loc[legs.v7_leg_status.eq("CANDIDATE")].copy()
    if qualifying.empty:
        return pd.DataFrame()
    # Refuse accidental mixing of sportsbooks/models/slates.
    needed = {"bookmaker","model","season","week","game_id","side","kind"}
    if not needed.issubset(qualifying):
        raise ValueError("Market-side audit missing segmentation columns")
    records=[]
    for (book,model,season,week), sub in qualifying.groupby(
            ["bookmaker","model","season","week"],dropna=False):
        options={}
        for game, choices in sub.groupby("game_id"):
            # Select at most one market alternative per game in an individual ticket.
            options[game]=choices.sort_values("conservative_ev",ascending=False).head(
                policy.max_sides_per_game).to_dict("records")
        for n in range(2,min(policy.max_legs,len(options))+1):
            for games in combinations(sorted(options),n):
                for selections in product(*(options[g] for g in games)):
                    p=prod(float(x["lower_probability"]) for x in selections)
                    d=prod(float(x["odds_decimal"])*(1-policy.price_stress)
                           for x in selections)
                    # Push settlement requires summing all win/push/loss combinations;
                    # reuse independent expected gross factor for expected value.
                    gross=prod(float(x["lower_probability"])*float(x["odds_decimal"])*
                               (1-policy.price_stress)+float(x.get("push_probability",0))
                               for x in selections)
                    ev=gross-1
                    records.append(dict(bookmaker=book,model=model,season=season,
                        week=week,legs=n,selections=" | ".join(
                            f'{x["game_id"]}:{x["side"]}:{x["kind"]}' for x in selections),
                        conservative_all_win_probability=p,conservative_ev=ev,
                        stressed_full_payout_decimal=d,
                        status=("RESEARCH_QUALIFIED" if
                           p>=policy.min_ticket_probability and ev>=policy.min_ticket_ev
                           else "REJECTED"),
                        caveat="Independence and compounded prices unverified; no accepted sportsbook parlay quote"))
    return pd.DataFrame(records).sort_values(
        ["conservative_ev","conservative_all_win_probability"],ascending=False)
