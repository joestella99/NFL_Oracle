"""Pregame-only quarterback EPA and opponent-aware efficiency features."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

ADVANCED_FEATURES = [
    'epa_offense_diff', 'epa_defense_diff', 'epa_matchup_diff',
    'qb_pass_epa_diff', 'qb_pass_share_diff',
]

def aggregate_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    needed={'game_id','posteam','defteam','epa','play_type','passer_player_id'}
    if not needed.issubset(pbp):
        raise ValueError(f'PBP missing required fields: {sorted(needed-set(pbp.columns))}')
    plays=pbp.loc[pbp['play_type'].isin(['pass','run'])].copy()
    plays['epa']=pd.to_numeric(plays['epa'],errors='coerce')
    plays=plays.dropna(subset=['game_id','posteam','defteam','epa'])
    if plays.empty:
        return pd.DataFrame(columns=['game_id','team','off_epa','def_epa','qb_epa','qb_share'])
    offense=plays.groupby(['game_id','posteam'],as_index=False).agg(
        off_epa=('epa','mean'), snaps=('epa','size')).rename(columns={'posteam':'team'})
    defense=plays.groupby(['game_id','defteam'],as_index=False)['epa'].mean().rename(
        columns={'defteam':'team','epa':'def_epa'})
    passing=plays.loc[(plays['play_type']=='pass') & plays['passer_player_id'].notna()].copy()
    if not passing.empty:
        passers=passing.groupby(['game_id','posteam','passer_player_id'],as_index=False).agg(
            pass_count=('epa','size'),pass_epa=('epa','mean'))
        passers=passers.sort_values(['game_id','posteam','pass_count','passer_player_id'],
            ascending=[True,True,False,True]).drop_duplicates(['game_id','posteam'])
        passers=passers.rename(columns={'posteam':'team','pass_epa':'qb_epa'})
        totals=passing.groupby(['game_id','posteam']).size().rename('total_passes').reset_index().rename(columns={'posteam':'team'})
        passers=passers.merge(totals,on=['game_id','team'])
        passers['qb_share']=passers['pass_count']/passers['total_passes']
    else:
        passers=pd.DataFrame(columns=['game_id','team','qb_epa','qb_share'])
    return offense.merge(defense,on=['game_id','team'],how='left').merge(
        passers[['game_id','team','qb_epa','qb_share']],on=['game_id','team'],how='left')

def make_pregame_advanced(games: pd.DataFrame, pbp: pd.DataFrame, window: int=8) -> pd.DataFrame:
    """Only prior game dates affect a pregame row; same-day games are isolated."""
    if window < 1: raise ValueError('window must be positive')
    summary=aggregate_pbp(pbp)
    lookup={(str(r.game_id),str(r.team)):r for r in summary.itertuples(index=False)}
    ordered=games[['game_id','gameday','home_team','away_team']].drop_duplicates('game_id').copy()
    ordered['gameday']=pd.to_datetime(ordered['gameday']).dt.normalize()
    history=defaultdict(lambda:deque(maxlen=window))
    def avg(team,key):
        nums=[v[key] for v in history[team] if np.isfinite(v[key])]
        return float(np.mean(nums)) if nums else 0.0
    out=[]
    for _,batch in ordered.sort_values(['gameday','game_id']).groupby('gameday',sort=True):
        for game in batch.itertuples(index=False):
            h,a=str(game.home_team),str(game.away_team)
            ho,ao=avg(h,'off'),avg(a,'off')
            hd,ad=avg(h,'def'),avg(a,'def')
            out.append({'game_id':game.game_id,'epa_offense_diff':ho-ao,
                'epa_defense_diff':ad-hd,'epa_matchup_diff':(ho-ad)-(ao-hd),
                'qb_pass_epa_diff':avg(h,'qb')-avg(a,'qb'),
                'qb_pass_share_diff':avg(h,'share')-avg(a,'share')})
        for game in batch.itertuples(index=False):
            for team in (str(game.home_team),str(game.away_team)):
                rec=lookup.get((str(game.game_id),team))
                if rec is not None:
                    def val(x): return float(x) if pd.notna(x) else np.nan
                    history[team].append({'off':val(rec.off_epa),'def':val(rec.def_epa),
                        'qb':val(rec.qb_epa),'share':val(rec.qb_share)})
    return pd.DataFrame(out)
