"""Validated, source-timestamped pregame personnel signals.

Input is an explicit as-of snapshot dataset, not postgame starters/participation.
Each observation MUST contain a trustworthy first-published timestamp.
Missing signals remain missing; no fabricated quarterback status.
"""
import pandas as pd
import numpy as np

COLUMNS=('game_id','team','position_group','status','observed_at','kickoff','source_url')
POSITIONS={'QB','OL'}
STATUS={'out','doubtful','questionable','available'}

def validate_personnel_snapshots(snapshots):
    missing=set(COLUMNS)-set(snapshots.columns)
    if missing:raise ValueError(f'Personnel snapshots missing {sorted(missing)}')
    d=snapshots.copy()
    d['observed_at']=pd.to_datetime(d.observed_at,utc=True,errors='raise')
    d['kickoff']=pd.to_datetime(d.kickoff,utc=True,errors='raise')
    if d[['game_id','team','position_group','status','source_url']].isna().any().any():
        raise ValueError('Personnel observations must include source and identity')
    if not d.position_group.isin(POSITIONS).all():raise ValueError('Unknown position group')
    if not d.status.isin(STATUS).all():raise ValueError('Unknown status')
    if not d.source_url.astype(str).str.startswith(('https://','http://')).all():
        raise ValueError('Missing verifiable source URL')
    if (d.observed_at>=d.kickoff).any():
        raise ValueError('Lookahead leakage: snapshot at or after kickoff')
    if d.duplicated(['game_id','team','position_group','observed_at']).any():
        raise ValueError('Conflicting duplicate snapshots')
    return d

def personnel_features(games,snapshots):
    d=validate_personnel_snapshots(snapshots)
    # Pick most recent independently timestamped report before kickoff.
    d=d.sort_values('observed_at').drop_duplicates(['game_id','team','position_group'],keep='last')
    rows=[]
    for g in games[['game_id','home_team','away_team']].itertuples(index=False):
        rec={'game_id':g.game_id}
        for side,team in (('home',g.home_team),('away',g.away_team)):
            for group in ('QB','OL'):
                matching=d.loc[(d.game_id==g.game_id)&(d.team==team)&(d.position_group==group)]
                status=matching.status.iloc[0] if len(matching) else None
                rec[f'{side}_{group.lower()}_out']=int(status=='out') if status is not None else np.nan
                rec[f'{side}_{group.lower()}_observed']=int(status is not None)
        rows.append(rec)
    out=pd.DataFrame(rows)
    for position in ('qb','ol'):
        out[f'{position}_out_diff']=out[f'away_{position}_out']-out[f'home_{position}_out']
    return out
