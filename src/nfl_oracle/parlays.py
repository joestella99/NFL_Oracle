"""Cross-game moneyline parlay research for *timestamped* weekly shortlist quotes.

Parlay decimal prices are products of each leg's decimal odds, as in the
Action Network parlay calculator. This is a mathematical payout calculation,
not confirmation the sportsbook would accept the legs simultaneously.

No same-game parlay inference. No synthetic bookmaker quotes. The probabilities
are conservative blended *research estimates*, not prospectively calibrated.
"""
from __future__ import annotations

from itertools import combinations
import math
import pandas as pd

from .core import ml_to_implied


PARLAY_WATCH_COLUMNS = [
    "bookmaker", "game_a", "selection_a", "game_b", "selection_b",
    "side_a", "side_b", "leg_a_odds", "leg_b_odds",
    "leg_a_probability", "leg_b_probability", "joint_probability_assuming_independence",
    "calculated_decimal_odds", "calculated_american_odds",
    "estimated_ev_assuming_independence", "suggested_extra_stake",
    "research_status", "captured_at_a", "captured_at_b",
]


def compounded_decimal(*odds: float) -> float:
    """Standard independent cross-game parlay payout, excluding original stake."""
    if len(odds) < 2:
        raise ValueError("At least two legs required")
    decimal = 1.0
    for price in odds:
        p = ml_to_implied(float(price))
        decimal *= (1.0 / p)
    return float(decimal)


def decimal_to_american(decimal: float) -> float:
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise ValueError("Invalid decimal price")
    return float((decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1))


def parlay_watchlist(
    quotes: pd.DataFrame,
    straight_picks: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    extra_stake_cap: float = 15.0,
    max_time_skew_minutes: int = 60,
    min_leg_probability: float = .60,
    min_joint_probability: float = .36,
    model_weight: float = .35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Review 2-leg ML parlays from straight-rejected games, same bookmaker.

    Finds at most one staked candidate across all sportsbooks, with every other
    examined pair available in the audit. No user-visible wager placed.
    Rejected-for-straights is based on actual ranking output, not an outcome.
    """
    required = {
        "game_id", "home_team", "away_team", "bookmaker", "captured_at",
        "home_moneyline", "away_moneyline", "model_home_prob",
    }
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"Missing quote columns: {sorted(missing)}")
    if not 0 <= model_weight <= 1 or extra_stake_cap < 0:
        raise ValueError("Invalid parlay research configuration")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone aware")
    time = pd.to_datetime(quotes.captured_at, utc=True, errors="raise")
    cutoff = cutoff.tz_convert("UTC")
    q = quotes.copy()
    q["captured_at"] = time
    q = q.loc[
        time.le(cutoff)
        & time.ge(cutoff - pd.Timedelta(hours=12))
        & q.bookmaker.notna()
    ].copy()
    if q.empty:
        return pd.DataFrame(columns=PARLAY_WATCH_COLUMNS), pd.DataFrame(columns=PARLAY_WATCH_COLUMNS)
    blocked = set(straight_picks.game_id.astype(str)) if "game_id" in straight_picks else set()
    q = q.loc[~q.game_id.astype(str).isin(blocked)]
    q = q.sort_values(["bookmaker", "game_id", "captured_at"], kind="stable").drop_duplicates(
        ["bookmaker", "game_id"], keep="last"
    )
    legs = []
    for g in q.itertuples(index=False):
        home_raw, away_raw = ml_to_implied(g.home_moneyline), ml_to_implied(g.away_moneyline)
        market_home = home_raw / (home_raw + away_raw)
        model_home = float(g.model_home_prob)
        if not math.isfinite(model_home) or not 0 <= model_home <= 1:
            continue
        home_p = (1 - model_weight) * market_home + model_weight * model_home
        home = home_p >= .5
        side_p = home_p if home else 1 - home_p
        if side_p < min_leg_probability:
            continue
        legs.append({
            "bookmaker": str(g.bookmaker),
            "game_id": str(g.game_id),
            "selection": str(g.home_team if home else g.away_team),
            "side": "home" if home else "away",
            "odds": float(g.home_moneyline if home else g.away_moneyline),
            "probability": side_p,
            "captured_at": g.captured_at,
        })
    reviewed = []
    for book, selections in pd.DataFrame(legs).groupby("bookmaker") if legs else []:
        for a, b in combinations(selections.to_dict("records"), 2):
            if a["game_id"] == b["game_id"]:
                continue
            skew = abs(a["captured_at"] - b["captured_at"])
            if skew > pd.Timedelta(minutes=max_time_skew_minutes):
                continue
            joint = a["probability"] * b["probability"]
            if joint < min_joint_probability:
                continue
            decimal = compounded_decimal(a["odds"], b["odds"])
            ev = joint * decimal - 1
            reviewed.append({
                "bookmaker": book, "game_a": a["game_id"], "selection_a": a["selection"],
                "game_b": b["game_id"], "selection_b": b["selection"],
                "side_a": a["side"], "side_b": b["side"],
                "leg_a_odds": a["odds"], "leg_b_odds": b["odds"],
                "leg_a_probability": a["probability"], "leg_b_probability": b["probability"],
                "joint_probability_assuming_independence": joint,
                "calculated_decimal_odds": decimal,
                "calculated_american_odds": decimal_to_american(decimal),
                "estimated_ev_assuming_independence": ev,
                "suggested_extra_stake": 0.0,
                "research_status": "WATCH_ONLY_UNVERIFIED_EDGE",
                "captured_at_a": a["captured_at"],
                "captured_at_b": b["captured_at"],
            })
    if not reviewed:
        return pd.DataFrame(columns=PARLAY_WATCH_COLUMNS), pd.DataFrame(columns=PARLAY_WATCH_COLUMNS)
    audit = pd.DataFrame(reviewed, columns=PARLAY_WATCH_COLUMNS).sort_values(
        ["estimated_ev_assuming_independence", "joint_probability_assuming_independence", "game_a", "game_b"],
        ascending=[False, False, True, True], kind="stable"
    ).reset_index(drop=True)
    best = audit.head(1).copy()
    r = best.iloc[0]
    if r.estimated_ev_assuming_independence > 0 and extra_stake_cap > 0:
        # This is a bounded EXPERIMENTAL amount, not a bet instruction.
        edge = min(1.0, float(r.estimated_ev_assuming_independence) / .15)
        confidence = min(1.0, max(0.0, (float(r.joint_probability_assuming_independence) - .36) / .40))
        best.loc[:, "suggested_extra_stake"] = round(min(extra_stake_cap, 2 + (extra_stake_cap - 2) * edge * confidence), 2)
        best.loc[:, "research_status"] = "EXPERIMENTAL_POSITIVE_ESTIMATED_EV_NOT_VALIDATED"
    return best, audit


def multi_leg_parlay_watchlist(
    quotes: pd.DataFrame,
    straight_picks: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    parlay_bankroll: float = 1000.,
    model_weight: float = .35,
    max_time_skew_minutes: int = 60,
    min_joint_probability: float = .10,
    probability_haircut_per_leg: float = .01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unlimited-count and unlimited-length cross-game round-robin research.

    Every 2..N subset is evaluated, including overlapping straight bets and
    other parlays. Bankroll, not arbitrary ticket counts, bounds total risk.
    Only conservatively positive-EV parlays receive research stake proposals.

    This excludes same-game market combinations and rejects stale, future or
    cross-book quote pairings. Probabilities assume cross-game independence.
    """
    if parlay_bankroll < 0 or not (0 <= model_weight <= 1):
        raise ValueError("Invalid portfolio risk configuration")
    required={"game_id","home_team","away_team","bookmaker","captured_at",
              "home_moneyline","away_moneyline","model_home_prob"}
    missing=required-set(quotes)
    if missing:raise ValueError(f"Missing quote columns: {sorted(missing)}")
    from itertools import combinations
    import numpy as np

    cols=["bookmaker","game_ids","selections","sides","leg_count",
          "leg_odds","leg_probabilities","joint_probability",
          "conservative_joint_probability","calculated_decimal_odds",
          "calculated_american_odds","estimated_ev","conservative_ev",
          "overlaps_straight","suggested_extra_stake","research_status",
          "earliest_quote","latest_quote"]
    cutoff=pd.Timestamp(as_of)
    if cutoff.tzinfo is None:raise ValueError("as_of must be timezone-aware")
    cutoff=cutoff.tz_convert("UTC")
    q=quotes.copy()
    q["captured_at"]=pd.to_datetime(q.captured_at,utc=True,errors="raise")
    q=q.loc[q.captured_at.le(cutoff) &
            q.captured_at.ge(cutoff-pd.Timedelta(hours=12)) &
            q.bookmaker.notna()].copy()
    if q.empty:return pd.DataFrame(columns=cols),pd.DataFrame(columns=cols)
    q=q.sort_values(["bookmaker","game_id","captured_at"],kind="stable").drop_duplicates(
        ["bookmaker","game_id"],keep="last")
    straight_games=set(straight_picks.game_id.astype(str)) if "game_id" in straight_picks else set()
    legs=[]
    for r in q.itertuples(index=False):
        try:
            home_raw=ml_to_implied(float(r.home_moneyline))
            away_raw=ml_to_implied(float(r.away_moneyline))
        except (TypeError,ValueError):
            continue
        base=home_raw/(home_raw+away_raw)
        p=float(r.model_home_prob)
        if not math.isfinite(p) or not 0<=p<=1:continue
        adjusted=(1-model_weight)*base+model_weight*p
        # Select one side of each game by conservative price-relative EV.
        side_options=[
            ("home",r.home_team,float(r.home_moneyline),adjusted),
            ("away",r.away_team,float(r.away_moneyline),1-adjusted),
        ]
        best=max(side_options,key=lambda x:
                 ((max(0.,x[3]-.01)/ml_to_implied(x[2]))-1,x[3]))
        legs.append(dict(game_id=str(r.game_id),bookmaker=str(r.bookmaker),
                         team=str(best[1]),side=best[0],odds=best[2],
                         p=best[3],captured_at=r.captured_at))
    rows=[]
    for book,grp in pd.DataFrame(legs).groupby("bookmaker") if legs else []:
        selections=grp.to_dict("records")
        for n in range(2,len(selections)+1):
            for group in combinations(selections,n):
                stamps=[r["captured_at"] for r in group]
                if max(stamps)-min(stamps)>pd.Timedelta(minutes=max_time_skew_minutes):
                    continue
                prob=math.prod(r["p"] for r in group)
                if prob<min_joint_probability:continue
                conserv=math.prod(max(0,r["p"]-probability_haircut_per_leg) for r in group)
                dec=math.prod(1/ml_to_implied(r["odds"]) for r in group)
                games=[r["game_id"] for r in group]
                if len(set(games))!=len(games):raise AssertionError("Repeated same game")
                rows.append({
                    "bookmaker":book,"game_ids":" | ".join(games),
                    "selections":" | ".join(r["team"] for r in group),
                    "sides":" | ".join(r["side"] for r in group),
                    "leg_count":n,"leg_odds":" | ".join(str(r["odds"]) for r in group),
                    "leg_probabilities":" | ".join(f'{r["p"]:.5f}' for r in group),
                    "joint_probability":prob,"conservative_joint_probability":conserv,
                    "calculated_decimal_odds":dec,
                    "calculated_american_odds":decimal_to_american(dec),
                    "estimated_ev":prob*dec-1,
                    "conservative_ev":conserv*dec-1,
                    "overlaps_straight":bool(set(games)&straight_games),
                    "suggested_extra_stake":0.,
                    "research_status":"WATCH_ONLY_NO_CONFIRMED_EDGE",
                    "earliest_quote":min(stamps),"latest_quote":max(stamps),
                })
    if not rows:return pd.DataFrame(columns=cols),pd.DataFrame(columns=cols)
    audit=pd.DataFrame(rows,columns=cols).sort_values(
        ["conservative_ev","joint_probability","game_ids"],ascending=[False,False,True],
        kind="stable").reset_index(drop=True)
    favorable=audit.loc[audit.conservative_ev>0].copy()
    if favorable.empty or parlay_bankroll==0:
        return pd.DataFrame(columns=cols),audit
    # Correlated Kelly estimates are NOT additive. Use the strongest ticket to
    # size a SINGLE aggregate bankroll sleeve, then split across all positive
    # opportunities. The parlay bankroll is an explicit user-configurable limit.
    top=favorable.iloc[0]
    individual_edge=float(top.conservative_ev)/(float(top.calculated_decimal_odds)-1)
    sleeve=min(parlay_bankroll,round(parlay_bankroll*.25*max(0,individual_edge),2))
    cents=round(sleeve*100)
    if cents==0:return pd.DataFrame(columns=cols),audit
    weights=np.maximum(favorable.conservative_ev.to_numpy(float),.001) * \
            np.maximum(favorable.conservative_joint_probability.to_numpy(float),.001)
    raw=weights/weights.sum()*cents
    pennies=np.floor(raw).astype(int)
    extra=cents-int(pennies.sum())
    if extra:pennies[np.argsort(-(raw-pennies),kind="stable")[:extra]]+=1
    favorable["suggested_extra_stake"]=pennies/100.
    favorable=favorable.loc[favorable.suggested_extra_stake>0].copy()
    favorable["research_status"]="EXPERIMENTAL_POSITIVE_ESTIMATED_EV"
    if favorable.suggested_extra_stake.sum()>parlay_bankroll+.005:
        raise AssertionError("Exceeded configured bankroll")
    return favorable,audit
