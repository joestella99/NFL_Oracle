"""V6 confidence-first research view alongside the frozen V5 value view.

Replay consumes previously generated out-of-sample leg audits; it never refits
on test results. Live use is integrated into parlay_live_mixed.py. No stakes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_oracle.conviction import (
    ConvictionPolicy, annotate_ticket, conviction_leg, leg_key,
    liked_pick_audit, rank_views,
)
from scripts.parlay_mixed_market_v5 import (
    VARIANTS, analyze_game, join_forecasts, load_policy as load_value_policy, ticket,
)

CONFIG = Path(__file__).resolve().parents[1] / "config/parlay_conviction_policy.json"


def load_policy(path: Path = CONFIG):
    config = json.loads(path.read_text())
    if config["status"] != "experimental_percentage_research":
        raise ValueError("Expected experimental conviction policy")
    policy = ConvictionPolicy(**config["thresholds"])
    policy.validate()
    return config["policy_id"], policy


def load_likes(path: str | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    rows = pd.read_csv(path, dtype=str)
    if not {"game_id", "side", "kind"}.issubset(rows):
        raise ValueError("Liked picks CSV needs game_id,side,kind")
    if rows[["game_id", "side", "kind"]].isna().any().any():
        raise ValueError("Liked picks cannot contain missing selections")
    if not rows.side.isin(["home", "away"]).all() or not rows.kind.isin(["ML", "ATS"]).all():
        raise ValueError("Liked picks require home/away and ML/ATS")
    return frozenset(leg_key(row) for row in rows.to_dict("records"))


def decorate_legs(rows: list[dict], policy: ConvictionPolicy) -> list[dict]:
    return [dict(row, selection=leg_key(row),
                 conviction_candidate=conviction_leg(row, policy)) for row in rows]


def enumerate_slate(rows: list[dict], value_policy, policy: ConvictionPolicy,
                    variant: str, liked=frozenset(), conviction_only=False) -> pd.DataFrame:
    choices = {}
    for row in rows:
        if row["conviction_candidate"] or (not conviction_only and row["candidate_selected"]):
            choices.setdefault(row["game_id"], []).append(row)
    records = []
    maximum = policy.max_legs if conviction_only else max(value_policy.max_legs, policy.max_legs)
    for length in range(2, min(maximum, len(choices)) + 1):
        for group in combinations(sorted(choices), length):
            for legs in product(*(choices[g] for g in group)):
                base = ticket(legs, value_policy, variant)
                # Additional conviction candidates must not silently alter
                # the frozen V5 value pool or its original leg-count rule.
                if length > value_policy.max_legs or not all(x["candidate_selected"] for x in legs):
                    base["status"] = "OUTSIDE_V5_CANDIDATE_POOL"
                records.append(annotate_ticket(base, legs, policy, liked))
    return rank_views(pd.DataFrame(records))


def build_slates(forecasts, margins, earlier_margins, value_policy,
                 policy, variants=VARIANTS, liked=frozenset()):
    policy.validate()
    games = join_forecasts(forecasts, margins, variants)
    prior = earlier_margins.loc[earlier_margins.season.lt(games.season.min())]
    errors = (prior.true_margin - prior.margin_prediction).to_numpy(float)
    errors = errors[np.isfinite(errors)]
    if len(errors) < 200:
        raise ValueError("At least 200 previous-season margin errors required")
    all_legs, all_tickets, weeks = [], [], []
    for variant in variants:
        for (season, week), slate in games.groupby(["season", "week"], sort=True):
            rows = []
            for game in slate.itertuples(index=False):
                rows.extend(decorate_legs(analyze_game(game, errors, variant, value_policy), policy))
            for row in rows:
                row["model"] = variant
            ranked = enumerate_slate(rows, value_policy, policy, variant, liked)
            all_legs.extend(rows)
            all_tickets.append(ranked)
            weeks.append(dict(season=int(season), week=int(week), model=variant,
                games=len(slate), combinations=len(ranked),
                qualifying_value=int(ranked.qualifies_value.sum()) if len(ranked) else 0,
                qualifying_conviction=int(ranked.qualifies_conviction.sum()) if len(ranked) else 0))
    return (pd.DataFrame(all_legs),
            pd.concat(all_tickets, ignore_index=True) if all_tickets else pd.DataFrame(),
            pd.DataFrame(weeks))


def replay(legs: pd.DataFrame, value_policy, policy: ConvictionPolicy):
    """Reuse frozen prior-year forecasts without ever ranking by outcomes."""
    if legs.duplicated(["season", "week", "model", "game_id", "kind", "side"]).any():
        raise ValueError("Duplicate market side in replay audit")
    blocks = []
    for (season, week, model), sub in legs.groupby(["season", "week", "model"], sort=True):
        rows = sub.rename(columns={"decimal_odds": "odds_decimal"}).to_dict("records")
        rows = decorate_legs(rows, policy)
        block = enumerate_slate(rows, value_policy, policy, model, conviction_only=True)
        if len(block):
            blocks.append(block.loc[block.qualifies_conviction])
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def summarize(tickets: pd.DataFrame, name: str) -> dict:
    if tickets.empty:
        return dict(view=name, tickets=0, equal_unit_roi=None)
    return dict(view=name, tickets=len(tickets),
                weeks=len(tickets[["season", "week"]].drop_duplicates()),
                hit_rate=float(tickets.true_unit_return.gt(0).mean()),
                mean_predicted_all_win_probability=float(tickets.estimated_all_win_probability.mean()),
                mean_model_ev=float(tickets.estimated_ev.mean()),
                equal_unit_roi=float(tickets.true_unit_return.mean()))


def comparison(conviction: pd.DataFrame, v5: pd.DataFrame):
    output = []
    annual = []
    for model in sorted(v5.model.unique()):
        c = conviction.loc[conviction.model.eq(model)] if len(conviction) else conviction
        v = v5.loc[v5.model.eq(model)]
        leader = v.sort_values(["conservative_ev", "estimated_all_win_probability", "selections"],
                    ascending=[False, False, True]).drop_duplicates(["season", "week"])
        views = {"v5_all_value": v, "v5_one_value_leader": leader,
                 "v6_all_conviction": c}
        if len(c):
            views.update(v6_one_confidence_leader=c.loc[c.conviction_rank.eq(1)],
                v6_stress_pass=c.loc[c.qualifies_value],
                v6_new_stress_fails=c.loc[~c.qualifies_value],
                v6_contains_low_edge=c.loc[c.contains_low_edge_leg])
        for name, data in views.items():
            output.append(dict(model=model, **summarize(data, name)))
            for season, sub in data.groupby("season"):
                annual.append(dict(model=model, season=int(season), **summarize(sub, name)))
    return pd.DataFrame(output), pd.DataFrame(annual)


def export(tickets: pd.DataFrame, path: Path):
    rows = tickets.copy()
    if len(rows) and "marginal_legs" in rows:
        rows["marginal_legs"] = rows.marginal_legs.map(json.dumps)
    rows.to_csv(path, index=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--leg-audit", required=True)
    cli.add_argument("--v5-tickets", required=True)
    cli.add_argument("--policy-config", default=str(CONFIG))
    cli.add_argument("--output", default="results/parlay_conviction_v6")
    args = cli.parse_args()
    policy_id, policy = load_policy(Path(args.policy_config))
    _, value_policy = load_value_policy()
    legs = pd.read_csv(args.leg_audit)
    v5 = pd.read_csv(args.v5_tickets)
    conviction = replay(legs, value_policy, policy)
    overall, annual = comparison(conviction, v5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    export(conviction, output / "conviction_tickets.csv")
    overall.to_csv(output / "comparison.csv", index=False)
    annual.to_csv(output / "annual.csv", index=False)
    report = dict(policy_id=policy_id, thresholds=asdict(policy),
        prior_v5_tickets=len(v5), conviction_tickets=len(conviction),
        input_audit="Frozen V5 out-of-sample market-side audit, including prior-year ATS residual probabilities.",
        interpretation="Historical equal-unit research; overlapping combinations, reused development seasons, unverified same-book compounded prices. No live-profit claim.")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
