"""Failure diagnostics and weekly block-bootstrap uncertainty for ROI engine v2.

Use full set of eligible weeks, including weeks with zero bets. Do not
re-optimize policy based on bootstrap results. Selected-vs-opponent feature
orientation is computed from PRE-GAME features only.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

SIGNALS=['epa_matchup_diff','qb_pass_epa_diff','run_attack_vs_defense',
         'epa_trimmed_diff','success_rate_diff','epa_std_diff']

def analyze(opportunities,bets,budget=100.,replicates=4000,seed=20261016):
    rng=np.random.default_rng(seed)
    confidence=[]
    by_season=[]
    signal_rates=[]
    for policy,eligible in opportunities.groupby('policy'):
        selected=bets[bets.policy.eq(policy)] if len(bets) else pd.DataFrame()
        for period,first,last in [('2021-2022',2021,2022),('2023-2025',2023,2025),('all',2021,2025)]:
            e=eligible[eligible.season.between(first,last)]
            s=selected[selected.season.between(first,last)] if len(selected) else pd.DataFrame()
            weeks=e[['season','week']].drop_duplicates()
            if len(weeks)==0:continue
            weekly=weeks.merge(s.groupby(['season','week']).agg(staked=('stake','sum'),
                profit=('profit','sum')).reset_index(),on=['season','week'],how='left').fillna(0.)
            arr=weekly[['profit','staked']].to_numpy(float)
            draw=rng.integers(0,len(weekly),(replicates,len(weekly)))
            samples=arr[draw].sum(axis=1)
            roi=np.divide(samples[:,0],samples[:,1],
                   out=np.full(replicates,np.nan),where=samples[:,1]>0)
            valid=roi[np.isfinite(roi)]
            confidence.append({
                'policy':policy,'period':period,'weeks':len(weekly),
                'roi_staked':float(s.profit.sum()/s.stake.sum()) if len(s) else None,
                'roi_bootstrap_5pct':float(np.quantile(valid,.05)) if len(valid) else None,
                'roi_bootstrap_95pct':float(np.quantile(valid,.95)) if len(valid) else None,
                'prob_positive_roi_bootstrap':float(np.mean(valid>0)) if len(valid) else None,
                'budget_utilization':float(weekly.staked.sum()/(len(weekly)*budget)),
                'budget_return':float(weekly.profit.sum()/(len(weekly)*budget)),
                'note':'Resampling observed weeks is descriptive uncertainty, not a future performance guarantee.'
            })
        for year,g in selected.groupby('season') if len(selected) else []:
            amt=float(g.stake.sum())
            by_season.append({'policy':policy,'season':int(year),'bets':len(g),
                'profit':float(g.profit.sum()),'stake':amt,'roi':float(g.profit.sum()/amt) if amt else None})
        for kind,group in selected.groupby('kind') if len(selected) else []:
            g=group.loc[~group.push]
            wins=g[g.won]
            missed=g[~g.won]
            direction=np.where(g.side.eq('home'),1.,-1.)
            for feature in SIGNALS:
                signed=g[feature].to_numpy(float)*direction
                mask=np.isfinite(signed)
                bad=signed[mask]<0
                y=g.won.to_numpy(bool)[mask]
                if y.any() and (~y).any():
                    signal_rates.append({
                        'policy':policy,'kind':kind,'signal':feature,
                        'wins':int(y.sum()),'misses':int((~y).sum()),
                        'opponent_advantage_in_misses':float(bad[~y].mean()),
                        'opponent_advantage_in_wins':float(bad[y].mean()),
                        'difference_miss_minus_win':float(bad[~y].mean()-bad[y].mean())
                    })
    return {'confidence':confidence,'limitations':[
        'These intervals resample past weeks and do not compensate for repeated historical strategy experimentation.',
        'Pre-game advantage frequency among misses must be compared with wins to avoid misleading correlations.',
        'A small positive historical ROI is not evidence of a market-beating algorithm.',
    ]},pd.DataFrame(by_season),pd.DataFrame(signal_rates)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--market-checks',required=True)
    ap.add_argument('--bets',required=True)
    ap.add_argument('--output',default='results/roi_stability')
    a=ap.parse_args()
    outcome,annual,signals=analyze(pd.read_csv(a.market_checks),pd.read_csv(a.bets))
    dst=Path(a.output);dst.mkdir(parents=True,exist_ok=True)
    (dst/'report.json').write_text(json.dumps(outcome,indent=2))
    annual.to_csv(dst/'annual.csv',index=False)
    signals.to_csv(dst/'miss_signal_vs_wins.csv',index=False)
    print(json.dumps(outcome,indent=2))
if __name__=='__main__':main()
