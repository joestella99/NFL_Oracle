"""Pregame-only offensive consistency features; no same-day/future results."""
from collections import defaultdict,deque
import numpy as np
import pandas as pd

STABILITY_FEATURES=['success_rate_diff','explosive_rate_diff','epa_trimmed_diff',
                    'epa_std_diff','negative_play_diff','pass_success_diff']

def make_stability(games,pbp,window=8):
    required={'game_id','posteam','epa','play_type'}
    if not required.issubset(pbp):raise ValueError(f'Missing PBP columns: {required-set(pbp)}')
    if window<1:raise ValueError('window must be positive')
    p=pbp.loc[pbp.play_type.isin(['run','pass']),['game_id','posteam','epa','play_type']].copy()
    p['epa']=pd.to_numeric(p.epa,errors='coerce')
    p=p.dropna(subset=['game_id','posteam','epa'])
    lookup={}
    for (gid,team),g in p.groupby(['game_id','posteam'],sort=False):
        e=g.epa.to_numpy(float)
        passed=g.loc[g.play_type=='pass','epa'].to_numpy(float)
        # Trimmed EPA reduces reliance on one unusually large play.
        trimmed=np.clip(e,np.quantile(e,.1),np.quantile(e,.9)).mean()
        lookup[(str(gid),str(team))]=dict(success=float(np.mean(e>0)),
          explosive=float(np.mean(e>1.5)),trimmed=float(trimmed),
          volatility=float(np.std(e)),negative=float(np.mean(e<-.5)),
          pass_success=float(np.mean(passed>0)) if len(passed) else 0.)
    order=games[['game_id','gameday','home_team','away_team','home_score','away_score']].drop_duplicates('game_id').copy()
    order['gameday']=pd.to_datetime(order.gameday).dt.normalize()
    hist=defaultdict(lambda:deque(maxlen=window))
    def avg(t,key):
        values=[v[key] for v in hist[t]]
        return float(np.mean(values)) if values else 0.
    out=[]
    for day,batch in order.sort_values(['gameday','game_id']).groupby('gameday',sort=True):
        for g in batch.itertuples(index=False):
            h,a=str(g.home_team),str(g.away_team)
            out.append(dict(game_id=g.game_id,success_rate_diff=avg(h,'success')-avg(a,'success'),
              explosive_rate_diff=avg(h,'explosive')-avg(a,'explosive'),
              epa_trimmed_diff=avg(h,'trimmed')-avg(a,'trimmed'),
              epa_std_diff=avg(h,'volatility')-avg(a,'volatility'),
              negative_play_diff=avg(a,'negative')-avg(h,'negative'),
              pass_success_diff=avg(h,'pass_success')-avg(a,'pass_success')))
        for g in batch.itertuples(index=False):
            if pd.isna(g.home_score) or pd.isna(g.away_score):continue
            for team in (str(g.home_team),str(g.away_team)):
                val=lookup.get((str(g.game_id),team))
                if val:hist[team].append(val)
    return pd.DataFrame(out)
