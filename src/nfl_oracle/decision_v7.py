"""V7 unified percentage-only straight and parlay research engine.

Calibrate only on completed OUT-OF-SAMPLE earlier seasons. Produce all four
ML/ATS sides, with a single probability source feeding straight and parlay
boards. No wagering, bankroll, ROI guarantees or automatic acceptance claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import isfinite, prod
import numpy as np
import pandas as pd

from nfl_oracle.core import ml_to_implied


@dataclass(frozen=True)
class V7Policy:
    max_legs: int = 3
    maximum_candidate_legs_per_game: int = 3
    min_candidate_ev: float = -0.025
    high_confidence_probability: float = 0.70
    stress_probability_points: float = 0.015
    disagreement_penalty: float = 0.10
    price_stress: float = 0.01
    min_raw_ticket_ev: float = 0.0
    min_robust_ticket_ev: float = 0.005
    max_quote_skew_minutes: int = 60
    max_featured_parlays_per_book_week: int = 5

    def validate(self):
        if not 2 <= self.max_legs <= 4 or not 1 <= self.maximum_candidate_legs_per_game <= 4:
            raise ValueError("Invalid ticket length or per-game candidate capacity")
        if not -1 < self.min_candidate_ev < 1 or not 0 <= self.high_confidence_probability <= 1:
            raise ValueError("Invalid candidate criteria")
        for name in ("stress_probability_points", "disagreement_penalty",
                     "price_stress", "min_raw_ticket_ev", "min_robust_ticket_ev"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError("Invalid percentage threshold: " + name)
        if self.max_quote_skew_minutes < 0 or self.max_featured_parlays_per_book_week < 1:
            raise ValueError("Invalid quote skew or display count")


@dataclass(frozen=True)
class Calibration:
    moneyline_weight: float
    margin_weight: float
    margin_residuals: tuple[float, ...]
    training_games: int
    last_training_season: int | None
    calibrated: bool

    @staticmethod
    def market_fallback(residuals=()):
        errors = tuple(float(x) for x in residuals if isfinite(float(x)))
        return Calibration(0.0, 0.0, errors, 0, None, False)


def _check_probability(p):
    x = float(p)
    if not isfinite(x) or not 0 <= x <= 1:
        raise ValueError("Missing or invalid probability")
    return x


def _margin_truth(frame):
    if "true_margin" in frame:
        return pd.to_numeric(frame.true_margin, errors="coerce").to_numpy(float)
    return (pd.to_numeric(frame.home_score, errors="coerce") -
            pd.to_numeric(frame.away_score, errors="coerce")).to_numpy(float)


def fit_calibration(history: pd.DataFrame, target_season: int,
                    model_col: str = "prediction", minimum_games: int = 200) -> Calibration:
    """Prior-year OOS fit: market anchored ML Brier and margin MSE weights.

    A previous game's actual margin is used ONLY for fitting calibration and
    constructing its earlier-season residual pool. The current season never
    trains its own forecast even if its results are present in the file.
    """
    columns = {"season", "game_id", "home_win", "market_home_prob",
               model_col, "margin_prediction", "spread_line"}
    if not columns.issubset(history):
        raise ValueError("Historical OOS calibration missing: " + str(sorted(columns - set(history))))
    before = history.loc[history.season.lt(target_season)].copy()
    if before.game_id.duplicated().any():
        raise ValueError("Duplicate OOS calibration game")
    if (before.season >= target_season).any():
        raise AssertionError("Future season in calibration")
    before = before.dropna(subset=["home_win", "market_home_prob", model_col,
                                  "margin_prediction", "spread_line"])
    if len(before) < minimum_games or before.season.nunique() < 2:
        return Calibration.market_fallback()
    y = before.home_win.to_numpy(float)
    market = before.market_home_prob.to_numpy(float)
    model = before[model_col].to_numpy(float)
    margin = before.margin_prediction.to_numpy(float)
    spread = before.spread_line.to_numpy(float)
    actual = _margin_truth(before)
    valid = (np.isfinite(y) & np.isfinite(market) & np.isfinite(model)
             & np.isfinite(margin) & np.isfinite(spread) & np.isfinite(actual)
             & (market > 0) & (market < 1) & (model > 0) & (model < 1))
    if valid.sum() < minimum_games:
        return Calibration.market_fallback()
    y, market, model, margin, spread, actual = (
        x[valid] for x in (y, market, model, margin, spread, actual))
    # Fixed grids, selected strictly on previous seasons. Mild zero-weight
    # regularization favors the sportsbook when football improvement is tiny.
    grid = np.linspace(0, 1, 21)
    wm = min(grid, key=lambda w: np.mean(((1-w)*market+w*model-y)**2) + .0005*w*w)
    # A handful of noisy historical wins is not evidence of model skill.
    # Require a meaningful absolute Brier improvement before trusting a
    # nonzero football weight instead of the sportsbook baseline.
    base_brier = float(np.mean((market-y)**2))
    best_brier = float(np.mean(((1-wm)*market+wm*model-y)**2))
    if base_brier - best_brier < .001:
        wm = 0.0
    ws = min(grid, key=lambda w: np.mean(((1-w)*spread+w*margin-actual)**2) + .05*w*w)
    adjusted = (1-ws)*spread + ws*margin
    residuals = tuple(float(x) for x in (actual - adjusted))
    return Calibration(float(wm), float(ws), residuals, int(len(y)),
                       int(before.season.max()), True)


def _decimal(american):
    try:
        price = ml_to_implied(float(american))
        return 1.0 / price
    except (TypeError, ValueError, OverflowError):
        return None


def _spread_probabilities(mean: float, line: float, residuals):
    errors = np.asarray(residuals, dtype=float)
    if len(errors) < 200 or not np.isfinite(errors).all():
        return None
    # NFL final scores are integer margins; integer spreads can push.
    spread_outcomes = np.rint(mean + errors) - line
    ph = float(np.mean(spread_outcomes > 1e-9))
    pa = float(np.mean(spread_outcomes < -1e-9))
    pp = float(np.mean(np.abs(spread_outcomes) <= 1e-9))
    return ph, pa, pp


def _leg_id(x):
    return f'{x["game_id"]}:{x["side"]}:{x["kind"]}'


def _settle_leg(kind, side, actual, line):
    if actual is None or not isfinite(float(actual)):
        return None
    edge = float(actual) if kind == "ML" else float(actual) - float(line)
    if abs(edge) < 1e-9:
        return "push"
    return "win" if (edge > 0) == (side == "home") else "loss"


def game_markets(game: dict, calibration: Calibration, policy: V7Policy):
    """Analyze BOTH sides of both markets; never force the ML favorite."""
    gid = str(game["game_id"])
    book = str(game.get("bookmaker", "HISTORICAL_UNVERIFIED"))
    hml, aml = _decimal(game.get("home_moneyline")), _decimal(game.get("away_moneyline"))
    hats, aats = _decimal(game.get("home_spread_odds")), _decimal(game.get("away_spread_odds"))
    if None in (hml, aml, hats, aats):
        return []
    market = (1/hml) / (1/hml + 1/aml)
    football = _check_probability(game.get("model_home_prob", market))
    home_p = (1-calibration.moneyline_weight)*market + calibration.moneyline_weight*football
    line = float(game["spread_line"])
    raw_margin = float(game.get("margin_prediction", line))
    mean = ((1-calibration.margin_weight)*line + calibration.margin_weight*raw_margin)
    covers = _spread_probabilities(mean, line, calibration.margin_residuals)
    ats_market = (1/hats) / (1/hats + 1/aats)
    truth = game.get("true_margin")
    asof = game.get("captured_at")
    result = []
    for kind, probs, prices, vig_market, push in (
        ("ML", (home_p, 1-home_p), (hml, aml), (market, 1-market), 0.0),
        ("ATS", (covers[:2] if covers else (None, None)), (hats, aats),
         (ats_market, 1-ats_market), covers[2] if covers else 0.0),
    ):
        for i, side in enumerate(("home", "away")):
            p = probs[i]
            if p is None:
                result.append(dict(game_id=gid, bookmaker=book, season=game.get("season"),
                    week=game.get("week"), kind=kind, side=side, selection=f"{gid}:{side}:{kind}",
                    status="MISSING_PRIOR_MARGIN_CALIBRATION", candidate=False,
                    liked=False, captured_at=asof))
                continue
            dec = prices[i]
            market_p = vig_market[i]
            stressed = max(0., float(p)-policy.stress_probability_points-
                           policy.disagreement_penalty*abs(p-market_p))
            ev = p*dec+push-1
            robust_ev = stressed*dec*(1-policy.price_stress)+push-1
            if robust_ev >= policy.min_robust_ticket_ev:
                status = "STRESS_POSITIVE_MODEL_EDGE"
            elif ev > 0:
                status = "SMALL_POSITIVE_MODEL_EDGE"
            elif p >= policy.high_confidence_probability:
                status = "HIGH_CONFIDENCE_BAD_PRICE"
            else:
                status = "NO_ESTIMATED_EDGE"
            result.append(dict(game_id=gid, bookmaker=book,
                season=game.get("season"), week=game.get("week"),
                kind=kind, side=side, selection=f"{gid}:{side}:{kind}",
                home_team=game.get("home_team"), away_team=game.get("away_team"),
                win_probability=float(p), push_probability=float(push),
                stressed_win_probability=float(stressed),
                market_no_vig_probability=float(market_p), decimal_odds=float(dec),
                american_odds=(game["home_moneyline"] if side=="home" else game["away_moneyline"]) if kind=="ML"
                    else (game["home_spread_odds"] if side=="home" else game["away_spread_odds"]),
                raw_ev=float(ev), stressed_ev=float(robust_ev),
                status=status, candidate=(ev >= policy.min_candidate_ev
                    or p >= policy.high_confidence_probability),
                captured_at=asof, actual_result=_settle_leg(kind,side,truth,line),
                liked=False, calibration_games=calibration.training_games,
                calibration_last_season=calibration.last_training_season,
                calibration_verified=calibration.calibrated))
    return result


def _candidate_options(legs: list[dict], policy: V7Policy, liked: frozenset = frozenset()):
    """Keep value and confident alternatives, not just the outright favorite."""
    options = [x for x in legs if x.get("candidate") and "raw_ev" in x]
    # Best expected-value ML, best expected-value ATS, and strongest favorite
    # may all be different selections; preserve each before trimming.
    chosen = {}
    for kind in ("ML", "ATS"):
        pool = [x for x in options if x["kind"] == kind]
        if pool:
            strongest_edge = max(pool, key=lambda x: (x["raw_ev"], x["win_probability"]))
            chosen[strongest_edge["selection"]] = strongest_edge
    if options:
        highest = max(options, key=lambda x: (x["win_probability"], x["raw_ev"]))
        chosen[highest["selection"]] = highest
    # Base search remains bounded, but eligible explicitly liked alternatives
    # are also enumerated rather than accidentally lost behind the top-three
    # optimization. Likes never bypass candidate admissibility or modify EV.
    base = sorted(chosen.values(), key=lambda x: (-x["raw_ev"], -x["win_probability"],
                                                  x["selection"]))[:policy.maximum_candidate_legs_per_game]
    included = {x["selection"]: x for x in base}
    for x in options:
        if x["selection"] in liked:
            included[x["selection"]] = x
    return list(included.values())


def _ticket(group, policy: V7Policy, liked: frozenset):
    """EV correct with spread pushes: E[gross] = product(p_win*price + p_push)."""
    assert len({x["game_id"] for x in group}) == len(group)
    prob = prod(x["win_probability"] for x in group)
    payout = prod(x["decimal_odds"] for x in group)
    raw = prod(x["win_probability"]*x["decimal_odds"] + x["push_probability"]
               for x in group) - 1
    robust = prod(x["stressed_win_probability"]*x["decimal_odds"]*
                  (1-policy.price_stress) + x["push_probability"]
                  for x in group) - 1
    outcomes = [x["actual_result"] for x in group]
    settled = None
    if all(x is not None for x in outcomes):
        realized = prod(0 if outcome=="loss" else
                        (1 if outcome=="push" else leg["decimal_odds"])
                        for leg,outcome in zip(group,outcomes))
        settled = realized - 1
    if robust >= policy.min_robust_ticket_ev:
        verdict = "STRESS_POSITIVE_MODEL_EDGE"
    elif raw > policy.min_raw_ticket_ev:
        verdict = "SMALL_POSITIVE_MODEL_EDGE"
    else:
        verdict = "NO_ESTIMATED_TICKET_EDGE"
    selections = [x["selection"] for x in group]
    return dict(bookmaker=group[0]["bookmaker"], season=group[0]["season"],
        week=group[0]["week"], selections=" | ".join(selections),
        game_ids=" | ".join(x["game_id"] for x in group), leg_count=len(group),
        market_mix="+".join(x["kind"] for x in group),
        full_payout_win_probability=float(prob), decimal_payout=float(payout),
        raw_ev=float(raw), stressed_ev=float(robust),
        mean_leg_win_probability=float(np.mean([x["win_probability"] for x in group])),
        liked_count=sum(s in liked for s in selections),
        contains_negative_edge_leg=any(x["raw_ev"] < 0 for x in group),
        contains_small_edge_leg=any(0 <= x["raw_ev"] < .02 for x in group),
        status=verdict, research_unit_return=settled, featured_rank=None)


def _rank_portfolio(tickets: pd.DataFrame, policy: V7Policy):
    if tickets.empty:
        return tickets
    result = tickets.copy()
    result["featured_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for _, sub in result.groupby(["season", "week", "bookmaker"], dropna=False):
        eligible = sub.loc[sub.raw_ev.gt(policy.min_raw_ticket_ev)].copy()
        used = []
        # Greedy diversification of overlapping tickets. Scores are for
        # presentation, not Kelly fractions or a simulated bankroll.
        while len(used) < policy.max_featured_parlays_per_book_week and not eligible.empty:
            def score(r):
                ids = set(r.game_ids.split(" | "))
                overlap = max((len(ids.intersection(u)) / len(ids.union(u))
                               for u in used), default=0.)
                # Utility preserves both small-edge hit rate and realistic
                # return; overlap is a ranking penalty, never a claim that
                # different games are statistically independent.
                return (.65*r.stressed_ev + .35*r.raw_ev
                        + .03*r.full_payout_win_probability
                        - .10*overlap)
            eligible["_score"] = eligible.apply(score, axis=1)
            best = eligible.sort_values(["_score", "full_payout_win_probability",
                                         "selections"], ascending=[False,False,True],
                                        kind="stable").iloc[0]
            result.loc[best.name, "featured_rank"] = len(used)+1
            used.append(set(best.game_ids.split(" | ")))
            eligible = eligible.drop(index=best.name)
    return result


def evaluate_slate(frame: pd.DataFrame, calibrations: dict[int, Calibration] | Calibration,
                   *, policy: V7Policy = V7Policy(), liked: frozenset = frozenset(),
                   live_asof=None):
    """Return complete straight and parlay audits (percentage-only).

    If live_asof is set, timestamps and future kickoff are MANDATORY.
    Historical rows are marked unverified regardless of quoted prices.
    """
    policy.validate()
    required = {"game_id","season","week","home_moneyline","away_moneyline",
                "home_spread_odds","away_spread_odds","spread_line"}
    if not required.issubset(frame):
        raise ValueError("Missing slate fields: "+str(sorted(required-set(frame))))
    data = frame.copy().reset_index(drop=True)
    if "bookmaker" not in data:
        data["bookmaker"] = "HISTORICAL_UNVERIFIED"
    if live_asof is not None:
        if "captured_at" not in data or "kickoff_utc" not in data:
            raise ValueError("Live evaluation requires quote capture and kickoff timestamps")
        asof = pd.Timestamp(live_asof)
        if asof.tzinfo is None:
            raise ValueError("As-of must contain timezone")
        qtime = pd.to_datetime(data.captured_at, utc=True, errors="coerce")
        kickoff = pd.to_datetime(data.kickoff_utc, utc=True, errors="coerce")
        asof = asof.tz_convert("UTC")
        data = data.loc[(qtime <= asof) &
            (qtime >= asof-pd.Timedelta(hours=12)) &
            (kickoff > asof) & (qtime < kickoff)].copy()
        data["captured_at"] = qtime.loc[data.index]
        if "true_margin" in data:
            data["true_margin"] = np.nan  # never expose future final scores
    if data.duplicated(["season","week","game_id","bookmaker"]).any():
        raise ValueError("Duplicate game/book/week quotes; deduplicate before evaluation")
    if not data.empty and data.bookmaker.isna().any():
        raise ValueError("Missing sportsbook name")
    straight = []
    for game in data.to_dict("records"):
        cal = calibrations[int(game["season"])] if isinstance(calibrations, dict) else calibrations
        straight.extend(game_markets(game, cal, policy))
    singles = pd.DataFrame(straight)
    if singles.empty:
        return singles, pd.DataFrame(), pd.DataFrame()
    singles["liked"] = singles.selection.isin(liked)
    singles["liked_rank"] = singles.liked.astype(int)
    singles = singles.sort_values(["season","week","bookmaker","stressed_ev",
                "win_probability"], ascending=[True,True,True,False,False],
                na_position="last",kind="stable").reset_index(drop=True)
    tickets = []
    for (season, week, book), sub in singles.groupby(["season","week","bookmaker"], dropna=False):
        grouped = {}
        for gid, legs in sub.groupby("game_id"):
            opts = _candidate_options(legs.to_dict("records"),policy,liked)
            if opts:
                grouped[gid] = opts
        for n in range(2, min(policy.max_legs,len(grouped))+1):
            for group in combinations(sorted(grouped),n):
                for legs in product(*(grouped[gid] for gid in group)):
                    if live_asof is not None:
                        stamps = [pd.Timestamp(x["captured_at"]) for x in legs]
                        if max(stamps)-min(stamps)>pd.Timedelta(minutes=policy.max_quote_skew_minutes):
                            continue
                    tickets.append(_ticket(legs,policy,liked))
    parlays = _rank_portfolio(pd.DataFrame(tickets),policy)
    if len(parlays):
        parlays["preferred_rank"] = pd.Series(pd.NA,index=parlays.index,dtype="Int64")
        for _, sub in parlays.groupby(["season","week","bookmaker"],dropna=False):
            ranked = sub.sort_values(["liked_count","full_payout_win_probability","raw_ev"],
                                     ascending=[False,False,False],kind="stable")
            parlays.loc[ranked.index,"preferred_rank"] = list(range(1,len(ranked)+1))
    # All explicitly liked picks remain visible even when not parlay eligible.
    liked_audit = singles.loc[singles.liked].copy()
    if len(liked_audit):
        picked = set()
        for t in tickets:
            picked.update(t["selections"].split(" | "))
        liked_audit["parlay_eligible"] = liked_audit.selection.isin(picked)
        liked_audit["explanation"] = np.where(liked_audit.parlay_eligible,
            "IN_ENUMERATED_TICKET",
            "NOT_IN_CANDIDATE_POOL_OR_NO_VALID_PARTNER")
    return singles, parlays, liked_audit
