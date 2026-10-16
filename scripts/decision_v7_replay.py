"""V7 2016-2025 chronological straight/ML+ATS parlay research comparison.

Consumes the existing OOS football predictions and historical margin replays.
Historical leg prices are mathematical proxies, NOT contemporaneous book slips.
No dollars, no automatic wager, no selection or hyperparameter sweep on test years.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from nfl_oracle.cli import get_data
from nfl_oracle.decision_v7 import V7Policy, fit_calibration, evaluate_slate
from scripts.parlay_v5_ten_year import margin_history


def joined_history(predictions, warmup, margins):
    raw = pd.concat([warmup, predictions], ignore_index=True)
    if raw.duplicated(["game_id","model"]).any():
        raise ValueError("Duplicate OOS football game/model")
    if margins.game_id.duplicated().any():
        raise ValueError("Duplicate OOS margin game")
    required = {"game_id","season","week","margin_prediction","true_margin",
                "spread_line","home_spread_odds","away_spread_odds"}
    if not required.issubset(margins):
        raise ValueError("Missing margin history fields")
    # Use OOS forecast prices, and never accidentally use future result data
    # to rank candidates. Outcome fields are retained strictly for settlement.
    raw = raw.drop(columns=["spread_line","home_spread_odds","away_spread_odds"],
                   errors="ignore")
    merged = raw.merge(margins[list(required)],on=["game_id","season","week"],
                       how="inner",validate="many_to_one")
    if merged.empty:
        raise ValueError("No common historical OOS forecasts")
    return merged


def _stat(d, label, predictor=""):
    if not len(d):
        return dict(view=label,tickets=0,roi_proxy=None,hit_rate=None,
                    predicted_win_rate=None)
    ret = d.research_unit_return.astype(float)
    p = d.full_payout_win_probability if predictor=="ticket" else None
    return dict(view=label,tickets=int(len(d)),
                roi_proxy=float(ret.mean()),hit_rate=float((ret>0).mean()),
                predicted_win_rate=float(p.mean()) if p is not None else None)


def replay(history: pd.DataFrame, *, first=2016, last=2025,
           models=("baseline","matchup","sustainability"),policy=V7Policy()):
    if last < first:
        raise ValueError("Invalid replay year window")
    all_singles, all_tickets, calibration_rows, summary, samples = [],[],[],[],[]
    for model in models:
        d=history.loc[history.model.eq(model)].copy()
        if d.empty:
            raise ValueError(f"No out-of-sample history for {model}")
        for season in range(first,last+1):
            current=d.loc[d.season.eq(season)].copy()
            if current.empty:
                continue
            cal=fit_calibration(d,season)
            if not cal.calibrated:
                raise ValueError(f"Missing >=200 OOS games from >=2 earlier seasons for {model} {season}")
            if cal.last_training_season>=season:
                raise AssertionError("Lookahead calibration")
            current["bookmaker"]="HISTORICAL_UNVERIFIED"
            current["model_home_prob"]=current.prediction
            singles,tickets,_=evaluate_slate(current,cal,policy=policy)
            if singles.empty:
                continue
            singles["model"]=model
            singles["historical_price_proxy"]=True
            if len(tickets):
                tickets["model"]=model
                tickets["historical_price_proxy"]=True
                all_tickets.append(tickets)
            all_singles.append(singles)
            y=current.home_win.astype(int).to_numpy()
            market=current.market_home_prob.to_numpy(float)
            model_prob=current.prediction.to_numpy(float)
            blended=market+cal.moneyline_weight*(model_prob-market)
            for label,p in (("market",market),("raw_football",model_prob),
                            ("v7_calibrated",blended)):
                calibration_rows.append(dict(model=model,season=season,view=label,
                   games=len(y),brier=float(np.mean((p-y)**2)),
                   logloss=float(np.mean(-y*np.log(np.clip(p,1e-8,1))-
                             (1-y)*np.log(np.clip(1-p,1e-8,1)))),
                   observed_home_win=float(y.mean()),
                   mean_predicted_home_win=float(p.mean())))
            samples.append(dict(model=model,season=season,training_games=cal.training_games,
                training_last_season=cal.last_training_season,
                ml_market_to_football_weight=cal.moneyline_weight,
                margin_market_to_football_weight=cal.margin_weight,
                prior_margin_residuals=len(cal.margin_residuals)))
    if not all_singles:
        raise ValueError("No OOS games for V7")
    singles=pd.concat(all_singles,ignore_index=True)
    parlays=pd.concat(all_tickets,ignore_index=True) if all_tickets else pd.DataFrame()
    for (model,season),one in singles.groupby(["model","season"]):
        # Exactly one straight per game, chosen before knowing results.
        viable=one.loc[one.raw_ev.gt(0)].sort_values(
            ["stressed_ev","raw_ev","win_probability","selection"],
            ascending=[False,False,False,True],kind="stable")
        chosen=viable.drop_duplicates("game_id")
        unit=np.where(chosen.actual_result.eq("win"),chosen.decimal_odds-1,
             np.where(chosen.actual_result.eq("push"),0.,-1.))
        summary.append(dict(model=model,season=int(season),view="straight_one_per_game",
           bets=len(chosen),unit_roi_proxy=float(unit.mean()) if len(unit) else None,
           hit_rate=float(np.mean(unit>0)) if len(unit) else None))
    if len(parlays):
        for (model,season),group in parlays.groupby(["model","season"]):
            # Compare independent weekly selection policies, not the huge,
            # highly overlapping 'all tickets' set as if each was independent.
            candidates=group.loc[group.raw_ev.gt(0)]
            for name,column in (("highest_confidence","full_payout_win_probability"),
                                ("highest_value","stressed_ev")):
                picked=(candidates.sort_values([column,"selections"],
                   ascending=[False,True],kind="stable")
                   .drop_duplicates(["season","week","bookmaker"]))
                row=_stat(picked,name,"ticket")
                summary.append(dict(model=model,season=int(season),view=name,
                   bets=row["tickets"],unit_roi_proxy=row["roi_proxy"],
                   hit_rate=row["hit_rate"],
                   predicted_win_rate=row["predicted_win_rate"]))
            featured=group.loc[group.featured_rank.eq(1)]
            row=_stat(featured,"diversified_leader","ticket")
            summary.append(dict(model=model,season=int(season),view="diversified_leader",
                bets=row["tickets"],unit_roi_proxy=row["roi_proxy"],
                hit_rate=row["hit_rate"],predicted_win_rate=row["predicted_win_rate"]))
    return (pd.DataFrame(summary),pd.DataFrame(calibration_rows),
            pd.DataFrame(samples),singles,parlays)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",required=True)
    parser.add_argument("--warmup",required=True)
    parser.add_argument("--recent-margins",required=True)
    parser.add_argument("--schedules-csv")
    parser.add_argument("--output",default="results/decision_v7")
    a=parser.parse_args()
    schedules=get_data(a.schedules_csv,list(range(2013,2026)))
    recent=pd.read_csv(a.recent_margins)
    margins=margin_history(schedules,recent)
    history=joined_history(pd.read_csv(a.predictions),pd.read_csv(a.warmup),margins)
    summary,calibration,weights,straight,tickets=replay(history)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for filename,frame in (
        ("annual.csv",summary),("calibration_by_year.csv",calibration),
        ("learned_prior_season_weights.csv",weights),
        ("straight_four_sides.csv",straight),
        ("parlay_candidates.csv",tickets),
        ("oos_calibration_inputs.csv",history.loc[history.model.eq("baseline")])):
        frame.to_csv(out/filename,index=False)
    aggregate=[]
    for (model,view),sub in summary.groupby(["model","view"]):
        n=int(sub.bets.sum())
        aggregate.append(dict(model=model,view=view,bets=n,
          mean_annual_proxy_roi=float(sub.unit_roi_proxy.mean()),
          years_with_picks=int(sub.unit_roi_proxy.notna().sum()),
          recent_2023_2025_mean_annual_roi=float(
              sub.loc[sub.season.ge(2023),"unit_roi_proxy"].mean())))
    report=dict(method="Preceding seasons OOS football/margin forecasts only; "
        "all four straight sides and diversified cross-game ML/ATS parlays",
        retrospective_proxies=aggregate,
        limitations=["Previously inspected seasons are development data.",
          "Individual NFLverse historical prices are NOT a verified same-book accepted parlay.",
          "Candidate tickets are highly overlapping and their returns are dependent.",
          "Market odds inputs cannot be independently confirmed pre-kickoff.",
          "Model weight fit and residuals are strictly before each evaluation season.",
          "No prospective profitability claim, dollars or wagering."])
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
