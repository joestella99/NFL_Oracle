"""Uniform market-override validation across separate NFL seasons.

Each season's weeks 1-5 predictions come from the same market-aware, 8-game-form
feature pipeline. Each week retrains on earlier seasons and previous weeks.
Select override threshold using ONLY the preceding season, then score this year.
No peeking at the evaluated season when choosing a threshold.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nfl_oracle.cli import get_data
from scripts.replay_weeks import walk_forward_predictions
from scripts.choose_winner_decisions import select_and_evaluate

def run(games, years, weeks=(1,2,3,4,5), pbp=None, include_matchups=False):
    years=sorted(set(map(int,years)))
    if len(years)<2: raise ValueError("At least two consecutive evaluation seasons needed")
    if any(b-a!=1 for a,b in zip(years,years[1:])):
        raise ValueError("Evaluation seasons must be consecutive")
    samples={}
    for year in years:
        samples[year]=walk_forward_predictions(games,year=year,weeks=list(weeks),pbp=pbp,include_matchups=include_matchups)
    rows=[]
    decisions=[]
    for year in years[1:]:
        selection=samples[year-1]
        test=samples[year]
        report,detail=select_and_evaluate(selection,test)
        e=report['evaluation']
        rows.append({'season':year,'selection_season':year-1,'games':e['market_correct']*0+len(test),
            'threshold':report['selected_threshold'],'oracle_correct':e['model_correct'],
            'market_correct':e['market_correct'],'decision_correct':e['selected_correct'],
            'market_accuracy':e['market_accuracy'],'decision_accuracy':e['selected_accuracy'],
            'overrides':e['overrides'],'correct_overrides':e['correct_overrides'],
            'incorrect_overrides':e['incorrect_overrides'],
            'market_brier':float(np.mean((test['market_home_prob']-test['home_win'])**2)),
            'model_brier':float(np.mean((test['model_home_prob']-test['home_win'])**2))})
        detail['test_season']=year
        decisions.append(detail)
    summary=pd.DataFrame(rows)
    all_decisions=pd.concat(decisions,ignore_index=True)
    total=int(summary.games.sum())
    market=int(summary.market_correct.sum())
    chosen=int(summary.decision_correct.sum())
    report={'evaluation_years':years[1:],'selection_years':years[:-1],
        'total_games':total,'market_correct':market,'decision_correct':chosen,
        'market_accuracy':market/total,'decision_accuracy':chosen/total,
        'net_extra_correct_picks':chosen-market,
        'seasons_improved':int((summary.decision_correct>summary.market_correct).sum()),
        'seasons_worse':int((summary.decision_correct<summary.market_correct).sum()),
        'seasons_equal':int((summary.decision_correct==summary.market_correct).sum()),
        'season_results':rows,
        'limitations':[
        'Uses one consistent feature schema across all seasons, with QB/EPA and matchup features when requested.',
        'Historical nflverse moneylines lack verified pregame timestamped executable prices.',
        'Evaluation is repeated prior-season selection, not proof of profitable wagers.',
        'No thresholds selected using the same season on which they are scored.'
        ]}
    return report,all_decisions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--years',nargs='+',type=int,default=[2022,2023,2024,2025,2026])
    p.add_argument('--train-start',type=int,default=2018)
    p.add_argument('--weeks',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--output',default='results/decision_multiseason')
    p.add_argument('--schedule-csv',default=None)
    p.add_argument('--with-pbp',action='store_true')
    p.add_argument('--matchups',action='store_true')
    p.add_argument('--pbp-csv',default=None)
    args=p.parse_args()
    if args.train_start>=min(args.years):p.error('Need pre-evaluation training seasons')
    data=get_data(args.schedule_csv,list(range(args.train_start,max(args.years)+1)))
    pbp=pd.read_csv(args.pbp_csv) if args.pbp_csv else None
    if (args.with_pbp or args.matchups) and pbp is None:
        import nflreadpy as nfl
        pbp=nfl.load_pbp(list(range(args.train_start,max(args.years)+1))).select(
            ['game_id','posteam','defteam','epa','play_type','passer_player_id']).to_pandas()
    report,details=run(data,args.years,args.weeks,pbp=pbp,include_matchups=args.matchups)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(report['season_results']).to_csv(out/'season_summary.csv',index=False)
    details.to_csv(out/'decisions.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
