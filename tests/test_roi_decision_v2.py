"""Decision engine checks: pushes, risk limits, conservative market gating."""
import numpy as np
import pandas as pd
from scripts.roi_decision_v2 import empirical_cover, market_probs, allocate, report, POLICIES

def test_discrete_margin_push_and_both_sides_cover():
    # 100 score margins exactly 7 and 100 exactly 10.
    h,a,p=empirical_cover(7,7,[0.]*100+[3.]*100)
    assert (h,a,p)==(.5,0.,.5)
    h,a,p=empirical_cover(7,7.5,[0.]*100+[3.]*100)
    assert (h,a,p)==(.5,.5,0.)

def test_no_vig_market_probabilities_sum_to_one():
    h,a=market_probs(-400,300)
    assert np.isclose(h+a,1.)
    assert np.isclose(h,.8/(.8+.25))

def test_no_bets_when_none_qualify():
    d=pd.DataFrame([dict(game_id='x',season=2025,week=1,policy='balanced',
                         qualifies=False,risk_score=-.1,estimated_ev=-.1)])
    assert allocate(d).empty

def test_one_exposure_per_game_and_partial_budget():
    rows=[]
    for game_id,kind,edge in [('a','ML',.09),('a','ATS',.07),('b','ATS',.04)]:
        rows.append(dict(game_id=game_id,season=2025,week=1,policy='balanced',
            kind=kind,side='home',qualifies=True,risk_score=edge,
            estimated_ev=edge,won=True,push=False,payout=.9))
    r=allocate(pd.DataFrame(rows))
    assert len(r)==2
    assert r.game_id.nunique()==2
    assert set(r.game_id)=={'a','b'}
    assert r.stake.sum()<100
    assert len(POLICIES)==2
