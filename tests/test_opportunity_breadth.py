import numpy as np
import pandas as pd
from scripts.opportunity_breadth import capped_weights, allocate_week

def test_waterfill_cap_and_exact_stake():
    v=capped_weights([10,1,1,1],10000,4000)
    assert np.isclose(sum(v),100.)
    assert max(v)<=40.
    assert v[0]==40.

def test_no_unnecessary_bet_cap_and_forced_breadth_label():
    rows=[]
    for i in range(8):
        rows.append(dict(policy="balanced",season=2025,week=1,
            game_id=f"g{i}",kind="ML",side="home",win_prob=.8,
            estimated_ev=.1-i*.02,risk_score=.1-i*.02,
            qualifies=i<5,payout=.25,won=True,push=False))
    out=allocate_week(pd.DataFrame(rows))
    assert len(out)==5
    assert np.isclose(out.stake.sum(),100.)
    assert out.stake.max()<=40.
    rows[1]["qualifies"]=False
    rows[2]["qualifies"]=False
    rows[3]["qualifies"]=False
    rows[4]["qualifies"]=False
    out=allocate_week(pd.DataFrame(rows))
    assert len(out)==3
    assert out.selected_below_gate.sum()==2
