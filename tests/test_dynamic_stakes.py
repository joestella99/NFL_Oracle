import numpy as np
import pandas as pd
from scripts.weighted_slate import simulate_weighted,distribute
from scripts.consensus_allocator import assemble

def fixture():
    rows=[]
    for i,(p,odds) in enumerate([(.9,-600),(.77,-145),(.7,-135),(.64,-150),(.60,-120),(.55,-110)]):
        for name in ['baseline','matchup','sustainability']:
            rows.append(dict(game_id=f'G{i}',season=2024,week=1,model=name,prediction=p,
              market_home_prob=p-.01,home_win=int(i%2==0),
              home_moneyline=odds,away_moneyline=110))
    return pd.DataFrame(rows)

def test_full_budget_and_differing_game_stakes():
    _,picks,weekly=simulate_weighted(fixture(),first=2024)
    chosen=picks[picks.stake>0]
    assert len(chosen)==6
    assert abs(chosen.stake.sum()-100)<.001
    assert chosen.stake.nunique()>2
    assert chosen.loc[chosen.game_id.eq('G0'),'stake'].iloc[0]>chosen.loc[chosen.game_id.eq('G5'),'stake'].iloc[0]

def test_dynamic_weights_change_with_price():
    base=assemble(fixture())
    before=distribute(base)
    changed=base.copy()
    changed.loc[changed.game_id.eq('G0'),'selected_odds']=-950
    after=distribute(changed)
    assert after.loc[after.game_id.eq('G0'),'stake'].iloc[0]<before.loc[before.game_id.eq('G0'),'stake'].iloc[0]
