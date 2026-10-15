"""Export conservative, timestamp-checked NFL injury report signals.

Official-ish historical nflverse injury reports contain date_modified, but this
may be the *last update*, not the first publication time. Rows edited after
kickoff are discarded, never backfilled. Injury listings are NOT confirmed
starters. Missing injury status is unknown, not healthy.
"""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from nfl_oracle.cli import get_data

OL={'T','OT','G','OG','C','OL','LT','RT','LG','RG'}

def extract(games,injuries):
    required={'game_id','season','week','home_team','away_team','gameday','gametime'}
    needed={'season','week','team','position','report_status','date_modified'}
    if not required.issubset(games):raise ValueError('Schedules need game date/time and IDs')
    if not needed.issubset(injuries):raise ValueError('Injury data lack report timestamp/status fields')
    schedule=games[games.game_type.eq('REG')].copy()
    local=(pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str),
                          errors='coerce').dt.tz_localize('America/New_York',
                          ambiguous='NaT',nonexistent='NaT'))
    schedule['kickoff_utc']=local.dt.tz_convert('UTC')
    reports=injuries.copy()
    reports['date_modified']=pd.to_datetime(reports.date_modified,utc=True,errors='coerce')
    reports['position']=reports.position.astype(str).str.upper()
    reports['report_status']=reports.report_status.astype(str).str.lower()
    reports['role']=np.where(reports.position.eq('QB'),'qb',
                        np.where(reports.position.isin(OL),'ol','other'))
    reports=reports[reports.role.isin(['qb','ol'])].copy()
    observations=[];audit=[]
    for game in schedule.itertuples(index=False):
        kickoff=game.kickoff_utc
        if pd.isna(kickoff):continue
        for team in (str(game.home_team),str(game.away_team)):
            sub=reports[(reports.season==game.season)&(reports.week==game.week)&(reports.team==team)]
            # A later overwrite is not evidence the last status was known before kickoff.
            known=sub[(sub.date_modified.notna())&(sub.date_modified<kickoff)].copy()
            latest=known.sort_values('date_modified').drop_duplicates(['position','team','season','week'],keep='last')
            for role in ['qb','ol']:
                available=latest[latest.role.eq(role)]
                row={'game_id':game.game_id,'team':team,'role':role,'kickoff_utc':str(kickoff),
                     'report_rows':len(available),
                     'out_count':int(available.report_status.eq('out').sum()),
                     'doubtful_count':int(available.report_status.eq('doubtful').sum()),
                     'questionable_count':int(available.report_status.eq('questionable').sum()),
                     'status_known':bool(len(available)>0)}
                observations.append(row)
            audit.append({'game_id':game.game_id,'team':team,'potential_reports':len(sub),
                          'reports_pre_kickoff':len(known),'reports_excluded_as_late':int(len(sub)-len(known))})
    return pd.DataFrame(observations),pd.DataFrame(audit)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seasons',type=int,nargs='+',default=[2023,2024,2025])
    p.add_argument('--output',default='results/personnel_reports')
    args=p.parse_args()
    import nflreadpy as nfl
    injuries=nfl.load_injuries(args.seasons).to_pandas()
    games=get_data(None,args.seasons)
    obs,audit=extract(games,injuries)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    obs.to_csv(out/'pregame_injury_counts.csv',index=False)
    audit.to_csv(out/'coverage_audit.csv',index=False)
    summary={'seasons':args.seasons,'available_team_role_rows':len(obs),
      'nonempty_report_rows':int(obs.status_known.sum()) if len(obs) else 0,
      'reports_excluded_late':int(audit.reports_excluded_as_late.sum()) if len(audit) else 0,
      'limitation':'An injury listing is not a verified starter or starting OL member. May be selection bias due to last-modified timestamps, historical fields must be audited before using as model feature.'}
    (out/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
