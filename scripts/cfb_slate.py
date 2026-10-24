"""College football slate scorer, isolated from NFL models.

Input CSV: game_id,kickoff_et,away,home,market,selection,american_odds,
           model_probability,probability_se
Markets: ml,spread,total. Each row represents an independently quoted outcome.
Only supplied, separately calibrated CFB model probabilities authorize picks.
Run: python scripts/cfb_slate.py --quotes data/cfb_2026_10_24_quotes.csv
"""
import argparse
import csv
import datetime as dt
import itertools
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def decimal_odds(american):
    x = float(american)
    if x == 0:
        raise ValueError("Zero is not valid American odds")
    return 1 + (x / 100 if x > 0 else 100 / abs(x))


def score(row, now, sigma=1.28, price_haircut=0.01):
    item = dict(row)
    kickoff = dt.datetime.fromisoformat(row["kickoff_et"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=ET)
    item["started"] = kickoff <= now
    item["decimal"] = decimal_odds(row["american_odds"])
    item["breakeven"] = 1 / item["decimal"]
    raw = row.get("model_probability", "").strip()
    se = row.get("probability_se", "").strip()
    if not raw or not se:
        item.update(status="UNSCORED_NO_CFB_MODEL", lower_probability=None, conservative_ev=None)
        return item
    p, err = float(raw), float(se)
    if not 0 <= p <= 1 or err < 0 or not math.isfinite(err):
        raise ValueError("Invalid probability or uncertainty")
    lower = max(0, p - sigma * err)
    stressed_decimal = 1 + (item["decimal"] - 1) * (1 - price_haircut)
    ev = lower * stressed_decimal - 1
    item.update(probability=p, lower_probability=lower, conservative_ev=ev,
                status=("STARTED" if item["started"] else
                        "QUALIFIED" if ev >= 0.02 else "REJECT_EV"))
    return item


def parlays(picks, max_legs=3):
    """Only independent GAME exposures; dependence between games remains unverified."""
    viable = [p for p in picks if p["status"] == "QUALIFIED"]
    results = []
    for length in range(2, min(max_legs, len(viable)) + 1):
        for combo in itertools.combinations(viable, length):
            if len({p["game_id"] for p in combo}) != length:
                continue
            # Multiplication is an independence approximation, NOT proof of fair parlay EV.
            prob = math.prod(p["lower_probability"] for p in combo)
            payout = math.prod(1 + (p["decimal"] - 1) * 0.99 for p in combo)
            ev = prob * payout - 1
            if ev >= 0.02 and prob >= 0.15:
                results.append({"legs": [p["selection"] for p in combo],
                                "game_ids": [p["game_id"] for p in combo],
                                "independence_approx_probability": round(prob, 4),
                                "conservative_ev_independence_approx": round(ev, 4),
                                "status": "RESEARCH_ONLY_UNVERIFIED_CORRELATION"})
    return sorted(results, key=lambda x: x["conservative_ev_independence_approx"], reverse=True)


def build(rows, now):
    picks = [score(r, now) for r in rows]
    return {"sport": "CFB", "as_of_et": now.isoformat(),
            "model": "external CFB probabilities required; NFL probabilities forbidden",
            "singles": sorted(picks, key=lambda p: (p["conservative_ev"] is not None,
                       p["conservative_ev"] if p["conservative_ev"] is not None else -100),
                       reverse=True),
            "parlays": parlays(picks),
            "warnings": ["No wagers without prior-to-kickoff, sport-specific calibrated probabilities.",
                         "Independence approximation is not a verified parlay edge.",
                         "Check actual bookmaker prices, injury changes, time stamps and local law."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    now = (dt.datetime.fromisoformat(args.as_of).astimezone(ET) if args.as_of
           else dt.datetime.now(ET))
    with args.quotes.open(newline="") as handle:
        report = build(list(csv.DictReader(handle)), now)
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
