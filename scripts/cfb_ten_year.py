"""Strict prior-season 2016-2025 CFB ML / ATS / parlay replay.

No sportsbook transactions. Archived quotes are NOT confirmed timestamped executable
pregame lines. ATS missing prices require a declared -110 hypothetical assumption.
Data version pinned to sportsdataverse/cfbfastR-data commit.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import re
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

SOURCE_SHA = "09f2595e1b23d485691577611d0d1f9cccbb4331"
ROOT = f"https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/{SOURCE_SHA}"
FIRST, LAST, WARMUP = 2016, 2025, 2014
# Fixed *before* seeing simulation outcomes; never select the best hindsight line.
BOOK_ORDER = ("PINNACLE", "Bovada", "BOVADA & bodog", "DraftKings",
              "FanDuel", "BetMGM", "Caesars", "bet365",
              "5Dimes & sportbet", "BETOnline & SPORTS BETTING")
EDGE_FLOOR = 0.02
PROB_STRESS = 0.02
PRICE_STRESS = 0.01
MARKET_WEIGHT = 0.65
ATS_MARKET_WEIGHT = 0.60


def get_csv(url, compression=None):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), compression=compression, low_memory=False, encoding_errors="replace")


def normalize(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""


def decimal_odds(v):
    x = float(v)
    if not math.isfinite(x) or x == 0:
        raise ValueError("Invalid American odds")
    return 1 + (x / 100 if x > 0 else 100 / abs(x))


def cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def load_schedules():
    years = []
    for year in range(WARMUP, LAST + 1):
        d = get_csv(f"{ROOT}/schedules/csv/cfb_schedules_{year}.csv")
        d = d[(d.home_division.str.lower() == "fbs") &
              (d.away_division.str.lower() == "fbs") &
              (d.season_type.str.lower() == "regular")].copy()
        for col in ["home_points", "away_points", "home_pregame_elo",
                    "away_pregame_elo", "game_id", "season", "week",
                    "home_id", "away_id"]:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["home_points", "away_points", "home_pregame_elo",
                             "away_pregame_elo", "game_id"])
        d = d[d.home_points != d.away_points].copy()
        d["game_id"] = d.game_id.astype("int64")
        d["season"] = d.season.astype("int64")
        d["week"] = d.week.astype("int64")
        d["home_id"] = d.home_id.astype("int64")
        d["away_id"] = d.away_id.astype("int64")
        d["home_margin"] = d.home_points - d.away_points
        d["home_win"] = (d.home_margin > 0).astype(int)
        d["home_field"] = (~d.neutral_site.astype(str).str.upper().eq("TRUE")).astype(int)
        d["elo_delta"] = (d.home_pregame_elo - d.away_pregame_elo) / 400.0
        years.append(d)
    games = pd.concat(years, ignore_index=True).sort_values(["season","week","start_date","game_id"])
    if games.game_id.duplicated().any():
        raise ValueError("Duplicated game IDs in source schedules")
    return games


def team_aliases(teams):
    aliases = {}
    for r in teams.itertuples(index=False):
        ids = getattr(r, "team_id", None)
        try:
            tid = int(ids)
        except (ValueError, TypeError):
            continue
        aliases[tid] = {normalize(getattr(r, field, "")) for field in
                        ("school", "abbreviation", "alt_name1", "alt_name2", "alt_name3")}
        aliases[tid].discard("")
    return aliases


def market_pairs(games, teams):
    """Select a predetermined book's two-sided quote, per game and market."""
    raw = get_csv(f"{ROOT}/betting/csv/cfb_line_odds.csv.gz", compression="gzip")
    raw = raw[raw.season.between(WARMUP, LAST) &
              raw.market_type.isin(["money_line", "spread"]) &
              raw.season_type.eq("regular")].copy()
    raw["game_id"] = pd.to_numeric(raw.game_id, errors="coerce")
    raw = raw[raw.game_id.notna()].copy()
    raw["game_id"] = raw.game_id.astype("int64")
    raw = raw[raw.game_id.isin(games.game_id)]
    aliases = team_aliases(teams)
    by_game = {}
    for r in games.itertuples():
        home = aliases.get(int(r.home_id), set()) | {normalize(r.home_team)}
        away = aliases.get(int(r.away_id), set()) | {normalize(r.away_team)}
        by_game[int(r.game_id)] = (home, away)
    def determine_side(gid, abbr):
        x = normalize(abbr)
        pair = by_game.get(int(gid))
        if not pair:
            return ""
        h, a = x in pair[0], x in pair[1]
        return "home" if h and not a else "away" if a and not h else ""
    raw["side"] = [determine_side(g, a) for g, a in zip(raw.game_id, raw.abbr)]
    unmatched = int((raw.side == "").sum())
    raw = raw[raw.side != ""].copy()
    raw["book_priority"] = raw.book.map({b: i for i, b in enumerate(BOOK_ORDER)}).fillna(1000)
    raw = raw.sort_values(["game_id", "market_type", "book_priority", "book"], kind="mergesort")
    # Don't cherry-pick a best historical price; first observed within each book-side pair.
    raw = raw.drop_duplicates(["game_id", "market_type", "book", "side"], keep="first")
    home = raw[raw.side == "home"]
    away = raw[raw.side == "away"]
    pairs = home.merge(away, on=["game_id", "market_type", "book"],
                       suffixes=("_home","_away"))
    pairs = pairs.sort_values(["game_id", "market_type", "book_priority_home", "book"],kind="mergesort")
    pairs = pairs.drop_duplicates(["game_id", "market_type"], keep="first")
    pairs = pairs[["game_id", "market_type", "book",
                   "odds_home", "odds_away", "lines_home", "lines_away"]]
    counts = {"raw_rows": int(len(raw) + unmatched), "unmatched_side_rows": unmatched,
              "matched_two_sided_market_pairs": int(len(pairs))}
    return pairs, counts


def fit_prior(games, season):
    train = games[games.season < season]
    if train.empty or train.season.max() >= season or train.season.min() > WARMUP:
        raise ValueError("Missing prior-only training sample")
    X = train[["elo_delta","home_field"]].to_numpy(dtype=float)
    logistic = LogisticRegression(C=10, max_iter=500, random_state=0).fit(X, train.home_win)
    ridge = Ridge(alpha=4.0).fit(X, train.home_margin)
    sigma = max(15.0, float(np.std(train.home_margin.to_numpy() - ridge.predict(X), ddof=1)))
    return logistic, ridge, sigma


def unit_result(won, push, odds):
    if push:
        return 0.0
    return decimal_odds(odds) - 1 if won else -1.0


def candidate(g, m, logistic, ridge, sigma):
    features = [[float(g.elo_delta), float(g.home_field)]]
    elo_home_win = float(logistic.predict_proba(features)[0][1])
    elo_margin = float(ridge.predict(features)[0])
    is_ml = m.market_type == "money_line"
    if is_ml:
        if pd.isna(m.odds_home) or pd.isna(m.odds_away):
            return []
        try:
            dh, da = decimal_odds(m.odds_home), decimal_odds(m.odds_away)
        except ValueError:
            return []
        imp_h, imp_a = 1/dh, 1/da
        market_home = imp_h / (imp_h + imp_a)
        home_p = MARKET_WEIGHT * market_home + (1 - MARKET_WEIGHT) * elo_home_win
        selections = [("home",home_p,float(m.odds_home),False),
                      ("away",1-home_p,float(m.odds_away),False)]
    else:
        if pd.isna(m.lines_home) or pd.isna(m.lines_away):
            return []
        line_h, line_a = float(m.lines_home), float(m.lines_away)
        if abs(line_h + line_a) > 0.1:
            return []
        pred = ATS_MARKET_WEIGHT * (-line_h) + (1 - ATS_MARKET_WEIGHT) * elo_margin
        home_p = cdf((pred + line_h) / sigma)
        # Missing spread odds are a hypothetical -110 flat assumption.
        oh = float(m.odds_home) if pd.notna(m.odds_home) and float(m.odds_home) != 0 else -110.0
        oa = float(m.odds_away) if pd.notna(m.odds_away) and float(m.odds_away) != 0 else -110.0
        selections = [("home",home_p,oh,pd.isna(m.odds_home)),
                      ("away",1-home_p,oa,pd.isna(m.odds_away))]
    result = []
    for side,p,odds,assumed in selections:
        dec = decimal_odds(odds)
        lower = max(0.0, p - PROB_STRESS)
        stressed_dec = 1 + (dec - 1) * (1 - PRICE_STRESS)
        ev = lower * stressed_dec - 1.0
        margin = float(g.home_margin) * (1 if side == "home" else -1)
        line = float(m.lines_home if side == "home" else m.lines_away) if not is_ml else 0.0
        adj_margin = margin + line
        won = adj_margin > 0
        push = (abs(adj_margin) < 1e-9) if not is_ml else False
        result.append({"game_id": int(g.game_id),"season":int(g.season),
          "week":int(g.week),"home_team":str(g.home_team),"away_team":str(g.away_team),
          "market":"ML" if is_ml else "ATS","selection":f"{str(g.home_team if side == 'home' else g.away_team)} {'ML' if is_ml else f'{line:+g}'}",
          "side":side,"book":str(m.book), "american_odds":odds,
          "assumed_spread_price":bool(assumed), "pred_probability":round(p,6),
          "stressed_probability":round(lower,6), "breakeven":round(1/dec,6),
          "conservative_ev":round(ev,6), "won":bool(won), "push":bool(push),
          "unit_profit":round(unit_result(won,push,odds),6),
          "qualified":bool(ev >= EDGE_FLOOR)})
    return result


def settle_parlay(legs):
    if any((not x["won"] and not x["push"]) for x in legs):
        return -1.0
    return float(np.prod([decimal_odds(x["american_odds"]) for x in legs if not x["push"]])) - 1.0


def make_parlays(singles, length):
    rows = []
    if singles.empty:
        return pd.DataFrame()
    for (season,week),group in singles.groupby(["season","week"]):
        # Fixed top-six per week: avoid explosion of hindsight-shared exposure.
        top = group.sort_values(["conservative_ev","game_id"], ascending=[False,True]).head(6)
        if len(top) < length:
            continue
        from itertools import combinations
        choices = []
        for combo in combinations(top.to_dict("records"),length):
            if len({x["game_id"] for x in combo}) != length:
                continue
            # A parlay must use a single quoted sportsbook for every leg.
            if len({x["book"] for x in combo}) != 1:
                continue
            p = math.prod(x["stressed_probability"] for x in combo)
            price = math.prod(1+(decimal_odds(x["american_odds"])-1)*(1-PRICE_STRESS) for x in combo)
            ev = p*price - 1
            if p < 0.15 or ev < EDGE_FLOOR:
                continue
            choices.append((ev, combo, p))
        if not choices:
            continue
        # One parlay of each length per week, selected without final scores.
        ev, legs, p = sorted(choices,key=lambda x:(-x[0],tuple(y["game_id"] for y in x[1])))[0]
        rows.append({"season":int(season),"week":int(week),"legs":length,
                     "selections":" | ".join(x["selection"] for x in legs),
                     "game_ids":" | ".join(str(x["game_id"]) for x in legs),
                     "joint_prob_independence_assumed":round(p,6),
                     "conservative_ev_independence_assumed":round(ev,6),
                     "unit_profit":round(settle_parlay(legs),6),
                     "won":bool(all(x["won"] or x["push"] for x in legs)),
                     "has_assumed_price":bool(any(x["assumed_spread_price"] for x in legs))})
    return pd.DataFrame(rows)


def summarize(df):
    if df.empty:
        return {"bets":0,"wins":0,"losses":0,"pushes":0,"units_profit":0,
                "roi":None,"hit_rate_decided":None}
    if "push" in df:
        push=df.push.astype(bool)
    else:
        push=pd.Series(False,index=df.index)
    wins = int((df.unit_profit > 0).sum())
    losses = int((df.unit_profit < 0).sum())
    count = int(len(df))
    return {"bets":count,"wins":wins,"losses":losses,"pushes":int(push.sum()),
            "units_profit":round(float(df.unit_profit.sum()),4),
            "roi":round(float(df.unit_profit.mean()),5),
            "hit_rate_decided":round(wins/(wins+losses),5) if wins+losses else None}


def run(out):
    games=load_schedules()
    teams=get_csv(f"{ROOT}/teams/teams_colors_logos.csv")
    odds,coverage=market_pairs(games,teams)
    game_records=games.set_index("game_id").to_dict("index")
    by_year={}
    candidates=[]
    training_rows={}
    for season in range(FIRST,LAST+1):
        logit,ridge,sigma=fit_prior(games,season)
        training_rows[season] = {"training_games":int((games.season<season).sum()),
                                 "margin_residual_sd":round(sigma,3)}
        season_markets=odds[odds.game_id.isin(games.loc[games.season==season,"game_id"])]
        for m in season_markets.itertuples(index=False):
            g=game_records.get(int(m.game_id))
            if g is None:
                continue
            # to named object via SimpleNamespace
            from types import SimpleNamespace
            candidates.extend(candidate(SimpleNamespace(**g,game_id=int(m.game_id)),
                                         m,logit,ridge,sigma))
    market_audit=pd.DataFrame(candidates)
    if market_audit.empty:
        raise RuntimeError("No market rows joined to 2016-25 scheduled FBS games.")
    positive=market_audit[market_audit.qualified].copy()
    # One model-selected leg per game (across ML and ATS), no same-game double dipping.
    singles=(positive.sort_values(["conservative_ev","market","side"],
                                  ascending=[False,True,True])
             .drop_duplicates(["game_id"],keep="first")
             .sort_values(["season","week","game_id"]))
    p2=make_parlays(singles,2)
    p3=make_parlays(singles,3)
    yearly=[]
    for y in range(FIRST,LAST+1):
        ss=singles[singles.season==y]
        year_games=games[games.season==y]
        row={"season":y,"fbs_games_with_ratings":int(len(year_games)),
             "market_side_candidates":int((market_audit.season==y).sum()),
             "eligible_single_games":int(len(ss))}
        for k,part in [("singles",ss),("ml",ss[ss.market=="ML"]),
                       ("ats",ss[ss.market=="ATS"]),
                       ("parlay2",p2[p2.season==y] if not p2.empty else p2),
                       ("parlay3",p3[p3.season==y] if not p3.empty else p3)]:
            d=summarize(part)
            row.update({f"{k}_{key}":value for key,value in d.items()})
        yearly.append(row)
    annual=pd.DataFrame(yearly)
    report={"title":"CFB 2016-2025 fixed-policy research replay","source_sha":SOURCE_SHA,
       "source": "sportsdataverse/cfbfastR-data schedules + archived betting odds",
       "research_only":True,
       "limitations":["Archived prices may not be timestamped pre-kickoff or executable.",
                      "Missing ATS juice assumed -110, not a historical quote.",
                      "CFBD pregame Elo is archive-derived; input snapshots not independently verified.",
                      "Parlay payouts multiply single-leg odds, not verified same-book parlay acceptances.",
                      "Model policy was designed after observing broader betting research; this is not untouched validation.",
                      "Preseason Elo inputs and absent matched quotes reduce sample coverage.",
                      "Risk stresses are preset heuristics, not calibrated confidence intervals."],
       "configuration":{"tested_seasons":list(range(FIRST,LAST+1)),
          "training_from":WARMUP,"price_assumption":"ATS -110 only when source price missing",
          "book_priority":BOOK_ORDER,"singles_per_game_max":1,
          "max_legs_for_weekly_parlay":3,"parlays_per_week_per_length":1,
          "edge_floor":EDGE_FLOOR,"probability_stress_absolute":PROB_STRESS,
          "profit_odds_haircut":PRICE_STRESS,"ml_blend_market_weight":MARKET_WEIGHT,
          "ats_blend_market_margin_weight":ATS_MARKET_WEIGHT},
       "source_coverage":coverage,
       "training_by_season":training_rows,
       "full_period":{"singles":summarize(singles),
                      "ml":summarize(singles[singles.market=="ML"]),
                      "ats":summarize(singles[singles.market=="ATS"]),
                      "parlay2":summarize(p2),"parlay3":summarize(p3)},
       "n_hypothetically_priced_ats_bets":int(singles.assumed_spread_price.sum()),
       "fbs_games_scored":int(len(games[games.season.between(FIRST,LAST)])),
       "audit_candidates":int(len(market_audit)),
       "n_weeks_with_any_bet":int(singles[["season","week"]].drop_duplicates().shape[0])}
    out.mkdir(exist_ok=True,parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    annual.to_csv(out/"annual.csv",index=False)
    singles.to_csv(out/"singles.csv",index=False)
    market_audit.to_csv(out/"market_audit.csv",index=False)
    if not p2.empty:p2.to_csv(out/"parlays_2.csv",index=False)
    if not p3.empty:p3.to_csv(out/"parlays_3.csv",index=False)
    print("CFB TEN-YEAR REPLAY",json.dumps(report["full_period"],sort_keys=True),flush=True)
    print("COVERAGE",json.dumps(report["source_coverage"],sort_keys=True),flush=True)
    print("YEARLY",annual[["season","fbs_games_with_ratings","eligible_single_games",
                           "singles_roi","ml_roi","ats_roi","parlay2_roi","parlay3_roi"]].to_json(orient="records"),flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/cfb_ten_year"))
    args=parser.parse_args()
    run(args.output)
