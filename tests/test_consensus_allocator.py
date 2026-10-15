import pandas as pd
from scripts.consensus_allocator import assemble,allocate,simulate

def sample(prob=.8,odd=-150):
    rows=[]
    for m in ['baseline','matchup','sustainability']:
        rows.append(dict(game_id='G',season=2024,week=1,model=m,prediction=prob,
          market_home_prob=.75,home_win=1,home_moneyline=odd,away_moneyline=130))
    return pd.DataFrame(rows)

def test_all_agree_but_bad_price_no_bet():
    d=allocate(assemble(sample(odd=-500)))
    assert d.stake.sum()==0

def test_disagreement_no_bet():
    s=sample()
    s.loc[s.model=='matchup','prediction']=.2
    assert allocate(assemble(s)).stake.sum()==0

def test_favorable_price_and_consensus_only_fractional_stake():
    d=allocate(assemble(sample()))
    assert d.stake.sum()>0
    assert d.stake.sum()<20
    assert d.profit.sum()>0

def test_allocation_never_spends_more_than_budget():
    d=pd.concat([sample().assign(game_id=str(i)) for i in range(10)],ignore_index=True)
    r,g,w=simulate(d,first=2024)
    assert g.stake.sum()<=20
    assert (w.unspent>=0).all()
