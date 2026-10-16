"""Predeclared side-by-side research on the same 2016-25 NFL Oracle weekly slates.

Parlay ticket prices are COMPOUNDED HISTORICAL LEG ODDS, not verified executable
same-book tickets. Ten seasons have already been inspected; this is exploratory,
NOT a clean holdout or evidence that any policy beats a sportsbook.
Selections never reference outcomes. Evaluate fixed risk, no-bet, parlay length,
overlap and bankroll allocation without cherry-picking winners.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

BUDGET = 100.0
POLICIES = (
    "checkpoint_extra_parlay", "straight_full", "straight_equal_stakes",
    "straight_abstain_fallback", "straight_quarter_risk",
    "straight_quarter_abstain", "two_leg_strict_ev", "two_leg_relaxed_ev",
    "two_leg_forced_5", "three_leg_forced_5", "four_leg_forced_5",
    "two_leg_forced_10", "two_leg_rejected_only", "two_leg_high_hit_prob",
    "two_leg_extra_5",
)

def decimal_odds(american):
    a = float(american)
    if not (a <= -100 or a >= 100):
        raise ValueError("Invalid American odds")
    return 1 + (a / 100 if a >= 100 else 100 / abs(a))

def find_ticket(slate, straight_games, leg_count=2, min_ev=None,
                rejected_only=False, ranking="conservative_ev"):
    # One side per game: highest model win probability; choose without outcomes.
    ml = slate[slate.kind.eq("ML")].sort_values(
        ["win_prob", "game_id", "side"], ascending=[False, True, True],
        kind="stable").drop_duplicates("game_id")
    ml = ml[ml.win_prob.ge(.60)]
    if rejected_only:
        ml = ml[~ml.game_id.isin(straight_games)]
    legs = ml.to_dict("records")
    if len(legs) < leg_count:
        return None
    best, best_key = None, None
    for combo in itertools.combinations(legs, leg_count):
        probability = float(np.prod([x["win_prob"] for x in combo]))
        conservative = float(np.prod([max(0., x["win_prob"] - .01) for x in combo]))
        dec = float(np.prod([decimal_odds(x["odds"]) for x in combo]))
        ev = probability * dec - 1
        conservative_ev = conservative * dec - 1
        if min_ev is not None and conservative_ev <= min_ev:
            continue
        if ranking == "joint_win_prob":
            key = (probability, conservative_ev)
        else:
            key = (conservative_ev, probability)
        if best_key is None or key > best_key:
            best_key = key
            best = dict(
                leg_count=leg_count, games=" | ".join(str(x["game_id"]) for x in combo),
                joint_win_probability=probability, calculated_decimal=dec,
                model_ev=ev, conservative_ev=conservative_ev,
                won=bool(all(x["won"] for x in combo)),
            )
    return best

def build(market, singles, original):
    rows = []
    pl = []
    group_cols = ["policy", "season", "week"]
    base = singles.groupby(group_cols, sort=True)
    market_index = market.groupby(group_cols, sort=True)
    original_lookup = {k:g for k,g in original.groupby(group_cols)} if len(original) else {}
    for key, s in base:
        slate = market_index.get_group(key)
        original_week = original_lookup.get(key)
        snet = float(s.profit.sum())
        fallback = bool(s.forced_fallback.any())
        games = set(s.game_id)
        eq = float(np.mean(np.where(s.push, 0., np.where(s.won, s.payout, -1.)))) * BUDGET
        existing_stake = float(original_week.stake.sum()) if original_week is not None else 0.
        existing_profit = float(original_week.profit.sum()) if original_week is not None else 0.
        existing_wins = int(original_week.both_won.sum()) if original_week is not None else 0
        combinations = {}
        for n in (2,3,4):
            combinations[n] = find_ticket(slate, games, leg_count=n)
        combinations["strict"] = find_ticket(slate, games, leg_count=2, min_ev=.0)
        combinations["relaxed"] = find_ticket(slate, games, leg_count=2, min_ev=-.05)
        combinations["rejected"] = find_ticket(slate, games, leg_count=2, rejected_only=True)
        combinations["high_prob"] = find_ticket(slate, games, leg_count=2, ranking="joint_win_prob")

        for strategy in POLICIES:
            straight_stake, straight_profit = BUDGET, snet
            ticket, ticket_stake, ticket_profit, ticket_stress = None, 0., 0., 0.
            hits = 0
            if strategy == "checkpoint_extra_parlay":
                ticket_stake, ticket_profit, hits = existing_stake, existing_profit, existing_wins
            elif strategy == "straight_equal_stakes":
                straight_profit = eq
            elif strategy == "straight_abstain_fallback" and fallback:
                straight_stake, straight_profit = 0., 0.
            elif strategy in ("straight_quarter_risk", "straight_quarter_abstain"):
                straight_stake, straight_profit = 25., snet * .25
                if strategy == "straight_quarter_abstain" and fallback:
                    straight_stake, straight_profit = 0., 0.
            elif strategy in (
                "two_leg_strict_ev", "two_leg_relaxed_ev", "two_leg_forced_5",
                "three_leg_forced_5", "four_leg_forced_5", "two_leg_forced_10",
                "two_leg_rejected_only", "two_leg_high_hit_prob", "two_leg_extra_5",
            ):
                which = {
                    "two_leg_strict_ev":"strict", "two_leg_relaxed_ev":"relaxed",
                    "two_leg_forced_5":2, "three_leg_forced_5":3,
                    "four_leg_forced_5":4, "two_leg_forced_10":2,
                    "two_leg_rejected_only":"rejected", "two_leg_high_hit_prob":"high_prob",
                    "two_leg_extra_5":2,
                }[strategy]
                ticket = combinations[which]
                if ticket is not None:
                    ticket_stake = 10. if strategy == "two_leg_forced_10" else 5.
                    if strategy != "two_leg_extra_5":
                        straight_stake = BUDGET-ticket_stake
                        straight_profit = snet * straight_stake/BUDGET
                    hits = int(ticket["won"])
                    ticket_profit = ticket_stake * (
                        ticket["calculated_decimal"]-1 if ticket["won"] else -1)
                    # A 2% haircut to the parlay's gross quoted payout.
                    ticket_stress = ticket_stake * (
                        max(0., ticket["calculated_decimal"]*.98-1) if ticket["won"] else -1)
                    pl.append(dict(strategy=strategy,policy=key[0],season=int(key[1]),
                        week=int(key[2]), stake=ticket_stake, profit=ticket_profit,
                        profit_quote_haircut_2pct=ticket_stress, **ticket))
            rows.append(dict(strategy=strategy, policy=key[0], season=int(key[1]),
                week=int(key[2]), forced_fallback=fallback,
                straight_staked=straight_stake, straight_profit=straight_profit,
                parlay_staked=ticket_stake, parlay_profit=ticket_profit,
                parlay_profit_2pct_worse_price=ticket_stress, parlay_hits=hits,
                total_staked=straight_stake+ticket_stake,
                total_profit=straight_profit+ticket_profit,
                stress_total_profit=straight_profit+ticket_stress
                    if strategy!="checkpoint_extra_parlay" else np.nan,
                ))
    return pd.DataFrame(rows), pd.DataFrame(pl)

def summarize(weekly):
    totals = []
    annual = []
    for (policy, strategy), frame in weekly.groupby(["policy","strategy"],sort=True):
        frame = frame.sort_values(["season","week"])
        n = len(frame)
        annual_profit = frame.groupby("season").total_profit.sum()
        path = np.r_[0., frame.total_profit.cumsum().to_numpy(float)]
        max_drawdown = float(np.max(np.maximum.accumulate(path)-path))
        stake = float(frame.total_staked.sum())
        profit = float(frame.total_profit.sum())
        parlay_staked = float(frame.parlay_staked.sum())
        parlay_profit = float(frame.parlay_profit.sum())
        totals.append(dict(policy=policy,strategy=strategy,weeks=n,
            parlay_tickets=int((frame.parlay_staked>0).sum()),
            parlay_wins=int(frame.parlay_hits.sum()),
            staked=round(stake,2),profit=round(profit,2),
            roi_pct=round(100*profit/stake,3) if stake else None,
            profit_per_available_budget_pct=round(100*profit/(BUDGET*n),3),
            max_drawdown_dollars=round(max_drawdown,2),
            losing_seasons=int((annual_profit<0).sum()),
            positive_weeks=int((frame.total_profit>0).sum()),
            parlay_staked=round(parlay_staked,2),
            parlay_profit=round(parlay_profit,2),
            parlay_roi_pct=round(100*parlay_profit/parlay_staked,3) if parlay_staked else None))
        for year,g in frame.groupby("season"):
            money = float(g.total_staked.sum()); gain=float(g.total_profit.sum())
            annual.append(dict(policy=policy,strategy=strategy,season=int(year),
                staked=round(money,2),profit=round(gain,2),
                roi_pct=round(100*gain/money,3) if money else None))
    return pd.DataFrame(totals),pd.DataFrame(annual)

def paired_resampling(weekly, n_draws=4000, seed=20261016):
    """Paired YEAR blocks, not independent tickets or hypothetical wins."""
    rng = np.random.default_rng(seed)
    out = []
    for policy,g in weekly.groupby("policy"):
        pivot = g.pivot_table(index=["season","week"],columns="strategy",values="total_profit")
        by_year = pivot.groupby(level="season").sum()
        base = by_year["straight_full"].to_numpy(float)
        years = by_year.index.to_numpy()
        draws = rng.integers(0,len(years),size=(n_draws,len(years)))
        for name in pivot.columns:
            diff = by_year[name].to_numpy(float)-base
            samples=diff[draws].sum(axis=1)
            out.append(dict(policy=policy,strategy=name,
                improvement_over_straight_dollars=round(float(diff.sum()),2),
                yearly_block_bootstrap_delta_5pct=round(float(np.quantile(samples,.05)),2),
                yearly_block_bootstrap_delta_95pct=round(float(np.quantile(samples,.95)),2),
                historical_resample_probability_beats_straights=round(float(np.mean(samples>0)),4),
                note="Exploratory uncertainty only; repeated historical inspection invalidates prospective CI claims."))
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-checks",required=True)
    ap.add_argument("--straight-bets",required=True)
    ap.add_argument("--old-parlays",required=True)
    ap.add_argument("--output",default="results/parlay_strategy_lab")
    args = ap.parse_args()
    market = pd.read_csv(args.market_checks)
    singles = pd.read_csv(args.straight_bets)
    original = pd.read_csv(args.old_parlays)
    weekly, tickets = build(market, singles, original)
    overall, annual = summarize(weekly)
    bootstrap = paired_resampling(weekly)
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    overall.to_csv(out/"overall.csv",index=False)
    annual.to_csv(out/"annual.csv",index=False)
    weekly.to_csv(out/"weekly.csv",index=False)
    tickets.to_csv(out/"generated_parlays.csv",index=False)
    bootstrap.to_csv(out/"paired_bootstrap.csv",index=False)
    results = {"tested_years":"2016-2025","historical_only":True,
        "baseline":"risk_tolerant straight_full; checkpoint adds original extra parlay",
        "pricing":"calculated cross-game historical odds; unverified simultaneous executable quotes",
        "budget":"$100 weekly baseline; diversion strategies keep $100 or less wagered; extra strategy risks $5 more",
        "selection":"Never based on outcomes; select max conservative adjusted EV among model-favorite tickets",
        "probability_haircut":"1 percentage point per leg",
        "limitations":["All ten historical seasons were previously inspected; not independent held-out verification.",
          "Actual bookmaker odds, simultaneous availability, parlay restrictions and fees are not verified.",
          "No adjustment for game correlations across tickets or strategy search multiple testing.",
          "Positive historic ROI and bootstrap support are not guarantees; any capital can be lost."],
        "risk_tolerant_results":overall.loc[overall.policy.eq("risk_tolerant")].to_dict("records")}
    (out/"report.json").write_text(json.dumps(results,indent=2))
    print("PARLAY STRATEGY LAB 2016-2025",flush=True)
    print(overall.to_string(index=False),flush=True)
    print("LATER PERIOD 2023-2025",flush=True)
    later=weekly[weekly.season.ge(2023)]
    l,_=summarize(later)
    print(l.to_string(index=False),flush=True)
    print("PAIRED YEAR-BLOCK BOOTSTRAP",flush=True)
    print(bootstrap.to_string(index=False),flush=True)

if __name__=="__main__":
    main()
