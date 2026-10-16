"""NFL Oracle parlay v2: independent football probability + exhaustive ticket portfolio.

Research-only. No wager is sent. Historical prices are mathematically compounded
and NOT verified simultaneously executable; outputs never establish a live edge.
No realized winner is used for picking, ranking, or allocating a ticket.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from math import prod, isfinite
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import logit

FEATURES = (
    "epa_offense_diff", "epa_defense_diff", "epa_matchup_diff",
    "qb_pass_epa_diff", "qb_pass_share_diff",
    "pass_attack_vs_defense", "run_attack_vs_defense",
    "pass_rate_diff", "success_rate_diff", "epa_trimmed_diff",
    "epa_std_diff", "negative_play_diff", "pass_success_diff",
)
MODELS = ("baseline", "matchup", "sustainability")


@dataclass(frozen=True)
class RiskThreshold:
    # Research defaults, not tuned or proven optimal.
    bankroll: float = 1000.
    min_leg_probability: float = .60
    min_joint_probability: float = .15
    min_conservative_ev: float = .02
    base_probability_haircut: float = .02
    disagreement_haircut_weight: float = .20
    per_leg_price_haircut: float = .02
    # V2 retains the strict individual-leg veto. V3 turns it off using config.
    require_positive_leg_ev: bool = True
    weekly_parlay_risk_fraction: float = .05
    maximum_ticket_risk_fraction: float = .005
    # V2: straight + parlay exposure in this cap. V3: parlay-only game cap.
    maximum_game_risk_fraction: float = .02
    # V3: an *additional* bound for straight + parlay risk on the same game.
    # None reproduces frozen v2 experimental semantics.
    maximum_total_game_risk_fraction: float | None = None
    minimum_ticket_stake: float = .25

    def validate(self):
        if not self.bankroll > 0: raise ValueError("bankroll must be positive")
        for field in (
            "min_leg_probability", "min_joint_probability",
            "base_probability_haircut", "disagreement_haircut_weight",
            "per_leg_price_haircut", "weekly_parlay_risk_fraction",
            "maximum_ticket_risk_fraction", "maximum_game_risk_fraction",
        ):
            value = getattr(self, field)
            if not 0 <= value < 1: raise ValueError("Invalid " + field)
        if self.maximum_total_game_risk_fraction is not None:
            if not 0 < self.maximum_total_game_risk_fraction <= 1:
                raise ValueError("Invalid maximum_total_game_risk_fraction")
        if not isinstance(self.require_positive_leg_ev,bool):
            raise ValueError("require_positive_leg_ev must be boolean")
        if self.minimum_ticket_stake < 0: raise ValueError("Invalid minimum stake")


def american_decimal(odds: float) -> float:
    price = float(odds)
    if not isfinite(price) or not (price <= -100 or price >= 100):
        raise ValueError("Invalid American moneyline")
    return 1 + (price / 100 if price > 0 else 100 / (-price))


def prepare_prediction_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Join pregame out-of-sample forecasts, game features, and historical labels.

    Output is ONE game row, not three correlated model rows.
    """
    required = {"game_id", "model", "season", "week", "prediction",
                "market_home_prob", "home_moneyline", "away_moneyline", "home_win"} | set(FEATURES)
    missing = required - set(raw)
    if missing: raise ValueError("Missing forecast fields: " + str(sorted(missing)))
    if raw.duplicated(["game_id", "model"]).any(): raise ValueError("Duplicate model/game")
    base = raw.loc[raw.model.eq("baseline"), [
        "game_id", "season", "week", "home_win", "market_home_prob",
        "home_moneyline", "away_moneyline", *FEATURES]].copy()
    wide = raw.pivot(index="game_id", columns="model", values="prediction").reset_index()
    if not set(MODELS).issubset(wide): raise ValueError("Missing model variants")
    table = base.merge(wide[["game_id", *MODELS]], on="game_id", validate="one_to_one")
    table = table.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    if table.empty: raise ValueError("No baseline games")
    pcols = ["market_home_prob", *MODELS]
    if not np.isfinite(table[pcols].to_numpy(float)).all(): raise ValueError("Invalid probabilities")
    if ((table[pcols] <= 0) | (table[pcols] >= 1)).any().any():
        raise ValueError("Probabilities must be inside (0,1)")
    return table


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Market anchoring PLUS independently learnable pregame football factors."""
    m = logit(np.clip(frame.market_home_prob.to_numpy(float), .001, .999))
    relative = [logit(np.clip(frame[col].to_numpy(float), .001, .999))-m for col in MODELS]
    numeric = frame[list(FEATURES)].to_numpy(float)
    numeric = np.where(np.isfinite(numeric), numeric, 0.)
    numeric = np.clip(numeric, -10, 10)
    return np.column_stack([m, *relative, numeric])


def football_probabilities(frame: pd.DataFrame, *, first_season: int = 2016) -> pd.DataFrame:
    """Train on STRICTLY PRIOR SEASONS' OOS predictions, never test outcomes.

    A new regularized model estimates win probability from market, ensemble
    disagreement, QB, offensive/defensive EPA, run/pass matchups, sustainability.
    Unavailable injuries and confirmed starters are NOT silently imputed.
    """
    f = frame.copy()
    f["football_home_prob"] = np.nan
    f["training_seasons"] = 0
    for year in sorted(f.season.unique()):
        if year < first_season: continue
        past = f.loc[f.season < year]
        target = f.loc[f.season.eq(year)]
        if past.season.nunique() < 2 or len(past) < 100 or target.empty:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.05, max_iter=1000, random_state=20261016),
        )
        model.fit(feature_matrix(past), past.home_win.astype(int))
        f.loc[target.index, "football_home_prob"] = model.predict_proba(feature_matrix(target))[:,1]
        f.loc[target.index, "training_seasons"] = past.season.nunique()
    return f


def candidate_legs(slate: pd.DataFrame, risk: RiskThreshold) -> list[dict]:
    """All modeled favorites, including games also used in straight bets.

    Selection is based on football probability, not whether a straight bet
    already uses that game. The bankroll allocator handles overlap.
    """
    out = []
    if slate.game_id.duplicated().any(): raise ValueError("One game per slate")
    for r in slate.itertuples(index=False):
        p_home = float(r.football_home_prob)
        market_home = float(r.market_home_prob)
        if not isfinite(p_home): continue
        home = p_home >= .5
        p = p_home if home else 1-p_home
        market_p = market_home if home else 1-market_home
        if p < risk.min_leg_probability: continue
        side = "home" if home else "away"
        odds = float(r.home_moneyline if home else r.away_moneyline)
        try: decimal = american_decimal(odds)
        except ValueError: continue
        divergence = np.std([float(getattr(r, col)) if home else 1-float(getattr(r,col))
                             for col in MODELS])
        # Conservative approximation, NOT a statistically validated confidence interval.
        lower = max(0., p-risk.base_probability_haircut
                    -risk.disagreement_haircut_weight*(abs(p-market_p)+divergence))
        out.append({
            "game_id": str(r.game_id), "side":side, "football_p":p,
            "market_p":market_p, "conservative_p":lower,
            "odds":odds, "decimal":decimal,
            "model_disagreement":float(divergence),
            "model_ev":p*decimal-1,
            "conservative_leg_ev":lower*decimal*(1-risk.per_leg_price_haircut)-1,
        })
    return out


def enumerate_tickets(legs: list[dict], risk: RiskThreshold) -> pd.DataFrame:
    """Every possible 2..N cross-game parlay, even when it does not qualify."""
    if len({x["game_id"] for x in legs}) != len(legs):
        raise ValueError("Repeated game in eligible legs")
    tickets=[]
    for size in range(2, len(legs)+1):
        for group in combinations(legs,size):
            ids=tuple(x["game_id"] for x in group)
            q=prod(x["conservative_p"] for x in group)
            p=prod(x["football_p"] for x in group)
            decimal=prod(x["decimal"] for x in group)
            executable_stress_decimal=decimal*(1-risk.per_leg_price_haircut)**size
            ev=p*decimal-1
            conservative_ev=q*executable_stress_decimal-1
            # V3 audits unfavorable individual legs without vetoing a
            # ticket that still passes the conservative *combined* EV test.
            # Legacy V2 keeps the individual-leg veto via a policy flag.
            weak_legs=sum(x["conservative_leg_ev"] <= 0 for x in group)
            status=("LEG_NOT_VALUE" if risk.require_positive_leg_ev and weak_legs else
                    "JOINT_PROBABILITY_BELOW_THRESHOLD" if q<risk.min_joint_probability else
                    "CONSERVATIVE_EV_BELOW_THRESHOLD" if conservative_ev<risk.min_conservative_ev else
                    "ELIGIBLE_RESEARCH")
            tickets.append({
                "game_ids":ids, "sides":tuple(x["side"] for x in group),
                "legs":size, "model_joint_probability":p,
                "risk_adjusted_joint_probability":q,
                "calculated_decimal_odds":decimal,
                "price_stressed_decimal_odds":executable_stress_decimal,
                "estimated_ev":ev,"conservative_ev":conservative_ev,
                "weak_leg_count":int(weak_legs),
                "status":status, "stake":0.,
            })
    return pd.DataFrame(tickets)


def allocate_tickets(tickets: pd.DataFrame, risk: RiskThreshold,
                     existing_straight_exposure: dict[str,float] | None = None) -> pd.DataFrame:
    """Many tickets may receive funds; correlated exposure uses shared caps.

    A game included in 7 tickets counts SEVEN times toward its exposure.
    Straight stake on that game reduces remaining room. Fixed cap per ticket
    prevents a few high-payout tickets from absorbing all parlay capital.
    """
    risk.validate()
    out=tickets.copy()
    if out.empty: return out
    exposure={str(g):float(x) for g,x in (existing_straight_exposure or {}).items()}
    if any(v<0 or not isfinite(v) for v in exposure.values()):
        raise ValueError("Invalid straight exposure")
    weekly_remaining=round(100*risk.bankroll*risk.weekly_parlay_risk_fraction)
    ticket_cap=round(100*risk.bankroll*risk.maximum_ticket_risk_fraction)
    game_cap=round(100*risk.bankroll*risk.maximum_game_risk_fraction)
    min_cents=max(1,round(100*risk.minimum_ticket_stake))
    occupied={g:round(100*v) for g,v in exposure.items()}
    # V3 reserves a separate game-level budget for parlay tickets while also
    # counting any straight bet against a larger combined per-game ceiling.
    separate_cap = risk.maximum_total_game_risk_fraction is not None
    parlay_occupied = {}
    combined_cap = (round(100*risk.bankroll*risk.maximum_total_game_risk_fraction)
                    if separate_cap else None)
    # Rank by conservative risk-adjusted Kelly signal and joint hit chance.
    eligible=out.loc[out.status.eq("ELIGIBLE_RESEARCH")].copy()
    eligible["ranking"]=eligible.apply(
        lambda r: r.conservative_ev/max(.001,r.price_stressed_decimal_odds-1)
                *np.sqrt(r.risk_adjusted_joint_probability),axis=1)
    eligible=eligible.sort_values(["ranking","risk_adjusted_joint_probability","legs"],
        ascending=[False,False,True],kind="stable")
    for idx,ticket in eligible.iterrows():
        if weekly_remaining < min_cents:break
        ids=ticket.game_ids
        if separate_cap:
            available=min(
                [weekly_remaining,ticket_cap]
                +[max(0,game_cap-parlay_occupied.get(g,0)) for g in ids]
                +[max(0,combined_cap-occupied.get(g,0)) for g in ids])
        else:
            available=min([weekly_remaining,ticket_cap]+
                [max(0,game_cap-occupied.get(g,0)) for g in ids])
        # Positive-EV Kelly is a total single-bet exposure guide, not additive
        # across correlated tickets. The sleeve/game caps dominate this guide.
        fractional_kelly=(ticket.conservative_ev/
                          max(.001,ticket.price_stressed_decimal_odds-1))*.25
        kelly_cap=round(100*risk.bankroll*max(0.,fractional_kelly))
        amount=min(available,kelly_cap)
        if amount<min_cents:continue
        out.at[idx,"stake"]=amount/100
        weekly_remaining-=amount
        for gid in ids:
            occupied[gid]=occupied.get(gid,0)+amount
            if separate_cap:
                parlay_occupied[gid]=parlay_occupied.get(gid,0)+amount
    if out.stake.sum() > risk.bankroll*risk.weekly_parlay_risk_fraction+.001:
        raise AssertionError("Weekly risk limit violated")
    if separate_cap:
        for gid,amount in parlay_occupied.items():
            if amount > game_cap:
                raise AssertionError("Parlay-only per-game exposure violated")
        for gid,amount in occupied.items():
            # The existing straight portfolio may already exceed the combined
            # cap: such positions cannot receive additional parlay exposure.
            if amount > max(combined_cap,round(100*exposure.get(gid,0))):
                raise AssertionError("Combined straight/parlay game cap violated")
    else:
        for gid,amount in occupied.items():
            if amount > game_cap and gid not in exposure:
                raise AssertionError("Game exposure limit violated")
            if gid in exposure and amount > max(game_cap,round(100*exposure[gid])):
                raise AssertionError("Exceeded per-game incremental risk cap")
    return out


def evaluate(raw_predictions: pd.DataFrame, straight: pd.DataFrame,
             original_parlays: pd.DataFrame, *,
             risk: RiskThreshold = RiskThreshold()) -> tuple[dict,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Same historical game outcomes for baseline and proposed parlay policy."""
    risk.validate()
    forecasts=football_probabilities(prepare_prediction_features(raw_predictions))
    eligible=forecasts.loc[forecasts.season.ge(2016)&forecasts.football_home_prob.notna()]
    outcomes=eligible.set_index("game_id").home_win.to_dict()
    joined_singles=straight.loc[straight.policy.eq("risk_tolerant")].copy()
    historic_parlays=original_parlays.loc[original_parlays.policy.eq("risk_tolerant")].copy()
    ss_lookup={k:g for k,g in joined_singles.groupby(["season","week"])}
    old_lookup={k:g for k,g in historic_parlays.groupby(["season","week"])}
    summary=[]; audit=[]
    for key,slate in eligible.groupby(["season","week"],sort=True):
        season,week=map(int,key)
        singles=ss_lookup.get(key)
        existing={}
        straight_profit=0.
        if singles is not None:
            straight_profit=float(singles.profit.sum())
            existing={str(k):float(v) for k,v in singles.groupby("game_id").stake.sum().items()}
        # Synthetic historical straights are included in per-game exposure.
        legs=candidate_legs(slate,risk)
        raw_tickets=enumerate_tickets(legs,risk)
        tickets=allocate_tickets(raw_tickets,risk,existing)
        for t in tickets.itertuples(index=False):
            won=all((bool(outcomes[gid]) if side=="home" else
                     not bool(outcomes[gid])) for gid,side in zip(t.game_ids,t.sides))
            profit=(t.calculated_decimal_odds-1)*t.stake if won else -t.stake
            audit.append({
                "season":season,"week":week,
                "games":" | ".join(t.game_ids),"sides":" | ".join(t.sides),
                "legs":t.legs,"status":t.status,
                "model_probability":t.model_joint_probability,
                "conservative_probability":t.risk_adjusted_joint_probability,
                "calculated_decimal_odds":t.calculated_decimal_odds,
                "estimated_ev":t.estimated_ev,"conservative_ev":t.conservative_ev,
                "weak_leg_count":t.weak_leg_count,
                "stake":t.stake,"won":won,"profit":profit,
            })
        week_audit=audit[-len(tickets):] if len(tickets) else []
        week_net=sum(t["profit"] for t in week_audit)
        week_staked=sum(t["stake"] for t in week_audit)
        old=old_lookup.get(key)
        old_p=float(old.profit.sum()) if old is not None else 0.
        old_stake=float(old.stake.sum()) if old is not None else 0.
        summary.append(dict(season=season,week=week,eligible_legs=len(legs),
                            examined=len(tickets),funded=sum(t["stake"]>0 for t in week_audit),
                            staked=week_staked,profit=week_net,
                            strict_combined_profit=straight_profit+week_net,
                            straight_profit=straight_profit,
                            checkpoint_profit=straight_profit+old_p,
                            checkpoint_parlay_profit=old_p,
                            checkpoint_parlay_staked=old_stake))
    week_df=pd.DataFrame(summary)
    audit_df=pd.DataFrame(audit)
    season_df=week_df.groupby("season").agg(
        weeks=("week","size"),tickets=("funded","sum"),staked=("staked","sum"),
        parlay_profit=("profit","sum"),straight_profit=("straight_profit","sum"),
        checkpoint_parlay_profit=("checkpoint_parlay_profit","sum"),
        checkpoint_parlay_staked=("checkpoint_parlay_staked","sum")).reset_index()
    amount=float(week_df.staked.sum())
    baseline=float(week_df.checkpoint_parlay_staked.sum())
    football_brier=float(np.mean((eligible.football_home_prob-eligible.home_win)**2))
    market_brier=float(np.mean((eligible.market_home_prob-eligible.home_win)**2))
    report={
        "research_only":True,
        "weeks":len(week_df),"prediction_games":len(eligible),
        "football_model_brier":football_brier,"market_brier":market_brier,
        "football_improves_market_brier":football_brier<market_brier,
        "all_ticket_candidates":len(audit_df),
        "funded_tickets":int((audit_df.stake>0).sum()) if len(audit_df) else 0,
        "model_parlay_staked":round(amount,2),
        "model_parlay_profit":round(float(week_df.profit.sum()),2),
        "model_parlay_roi":float(week_df.profit.sum()/amount) if amount else None,
        "checkpoint_parlay_staked":round(baseline,2),
        "checkpoint_parlay_profit":round(float(week_df.checkpoint_parlay_profit.sum()),2),
        "checkpoint_parlay_roi":float(week_df.checkpoint_parlay_profit.sum()/baseline) if baseline else None,
        "risk_defaults":risk.__dict__,
        "limitations":[
            "2016-2025 were repeatedly inspected and are not a clean prospective holdout.",
            "NFLverse historical prices cannot prove simultaneous same-book executable offers.",
            "Cross-game independence is only an approximation; common prediction errors can correlate tickets.",
            "Football features cover observed pregame historical QB/EPA/matchups/consistency, not live injuries or verified starting lineups.",
            "The conservative probability discount is heuristic, NOT a certified confidence bound.",
            "An empty qualifying portfolio is valid and preferable to wagering negative estimated EV.",
        ],
    }
    return report,week_df,season_df,audit_df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--warmup",required=True)
    ap.add_argument("--straight-bets",required=True)
    ap.add_argument("--old-parlays",required=True)
    ap.add_argument("--output",default="results/parlay_portfolio_v2")
    a=ap.parse_args()
    all_predictions=pd.concat([pd.read_csv(a.warmup),pd.read_csv(a.predictions)],
                              ignore_index=True)
    report,weekly,annual,audit=evaluate(all_predictions,pd.read_csv(a.straight_bets),
        pd.read_csv(a.old_parlays))
    dst=Path(a.output);dst.mkdir(parents=True,exist_ok=True)
    (dst/"report.json").write_text(json.dumps(report,indent=2))
    weekly.to_csv(dst/"weekly.csv",index=False)
    annual.to_csv(dst/"annual.csv",index=False)
    audit.to_csv(dst/"all_combinations_and_exposure.csv",index=False)
    print("MULTI-TICKET PARLAY ENGINE V2",flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print(annual.to_string(index=False),flush=True)

if __name__=="__main__":
    main()
