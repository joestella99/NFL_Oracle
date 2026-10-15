"""Pregame scheme-specific matchup features from completed prior NFL plays.

Calculated without final outcomes from the same day. Uses pass/run EPA and volume,
not unavailable pressure or verified injury data. Opponent-relative values compare
performance with the rival's defense/offense PRIOR to that game.
"""
from collections import defaultdict,deque
import numpy as np
import pandas as pd

MATCHUP_FEATURES=[
 'pass_attack_edge','run_attack_edge','pass_defense_edge','run_defense_edge',
 'pass_rate_diff','pass_attack_vs_defense','run_attack_vs_defense',
 'pass_share_matchup','offense_residual_diff','defense_residual_diff',
]
def matchup_features(games:pd.DataFrame,pbp:pd.DataFrame,window:int=8)->pd.DataFrame:
    needed={'game_id','posteam','defteam','epa','play_type'}
    if not needed.issubset(pbp):raise ValueError(f'Missing PBP columns {sorted(needed-set(pbp))}')
    if window<=0:raise ValueError('window must be positive')
    plays=pbp.loc[pbp.play_type.isin(['pass','run']),list(needed)].copy()
    plays['epa']=pd.to_numeric(plays.epa,errors='coerce')
    plays=plays.dropna(subset=['game_id','posteam','defteam','epa'])
    grouped=plays.groupby(['game_id','posteam','play_type']).agg(mean_epa=('epa','mean'),plays=('epa','size')).reset_index()
    lookup={(str(x.game_id),str(x.posteam),x.play_type):(float(x.mean_epa),int(x.plays)) for x in grouped.itertuples(index=False)}
    order=games[['game_id','gameday','home_team','away_team','home_score','away_score']].drop_duplicates('game_id').copy()
    order['gameday']=pd.to_datetime(order.gameday).dt.normalize()
    history=defaultdict(lambda:deque(maxlen=window))
    def avg(team,k):
        v=[record[k] for record in history[team] if np.isfinite(record[k])]
        return float(np.mean(v)) if v else 0.
    result=[]
    for _,batch in order.sort_values(['gameday','game_id']).groupby('gameday',sort=True):
        for game in batch.itertuples(index=False):
            h,a=str(game.home_team),str(game.away_team)
            hp,ap=avg(h,'pass_off'),avg(a,'pass_off')
            hr,ar=avg(h,'run_off'),avg(a,'run_off')
            hd,ad=avg(h,'pass_def'),avg(a,'pass_def')
            hdr,adr=avg(h,'run_def'),avg(a,'run_def')
            hs,aas=avg(h,'pass_share'),avg(a,'pass_share')
            result.append({'game_id':game.game_id,
                'pass_attack_edge':hp-ap,'run_attack_edge':hr-ar,
                'pass_defense_edge':ad-hd,'run_defense_edge':adr-hdr,
                'pass_rate_diff':hs-aas,
                'pass_attack_vs_defense':(hp-ad)-(ap-hd),
                'run_attack_vs_defense':(hr-adr)-(ar-hdr),
                'pass_share_matchup':(hs-avg(a,'opp_pass_share'))-(aas-avg(h,'opp_pass_share')),
                'offense_residual_diff':avg(h,'off_residual')-avg(a,'off_residual'),
                'defense_residual_diff':avg(a,'def_residual')-avg(h,'def_residual')})
        # Update only AFTER every game that day has been assigned its features.
        for game in batch.itertuples(index=False):
            if pd.isna(game.home_score) or pd.isna(game.away_score):continue
            h,a=str(game.home_team),str(game.away_team)
            data={}
            for team,rival in [(h,a),(a,h)]:
                pr,pn=lookup.get((str(game.game_id),team,'pass'),(np.nan,0))
                rr,rn=lookup.get((str(game.game_id),team,'run'),(np.nan,0))
                dr,_=lookup.get((str(game.game_id),rival,'pass'),(np.nan,0))
                drr,_=lookup.get((str(game.game_id),rival,'run'),(np.nan,0))
                n=pn+rn
                data[team]={'pass_off':pr,'run_off':rr,'pass_def':dr,'run_def':drr,
                    'pass_share':pn/n if n else np.nan,
                    'opp_pass_share':None,
                    'off_residual':np.nan,'def_residual':np.nan}
            for team,rival in [(h,a),(a,h)]:
                d=data[team]
                d['opp_pass_share']=data[rival]['pass_share']
                # Expected EPA based on rival's PREVIOUS defensive/offensive form.
                opp_expected=avg(rival,'pass_def')
                own_expected=avg(team,'pass_def')
                d['off_residual']=d['pass_off']-opp_expected if np.isfinite(d['pass_off']) else np.nan
                d['def_residual']=data[rival]['pass_off']-own_expected if np.isfinite(data[rival]['pass_off']) else np.nan
            for team in (h,a):
                if np.isfinite(data[team]['pass_off']) or np.isfinite(data[team]['run_off']):
                    history[team].append(data[team])
    return pd.DataFrame(result)
