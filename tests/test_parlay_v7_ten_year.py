import pandas as pd
from scripts.parlay_v7_ten_year import run
from nfl_oracle.v7 import V7Policy


def sample():
    rows=[]
    for i in range(40):
        rows.append(dict(season=2025,week=1,game_id=f"old{i}",model="football",
            kind="ML",side="home",win_probability=.75,odds_decimal=1.6,
            push_probability=0.,won=i%4!=0,pushed=False))
    for id in ("newA","newB"):
        rows.append(dict(season=2026,week=2,game_id=id,model="football",
            kind="ML",side="home",win_probability=.75,odds_decimal=1.6,
            push_probability=0.,won=id=="newA",pushed=False))
    return pd.DataFrame(rows)


def test_walkforward_tickets_have_real_settlements_and_no_future_training():
    policy=V7Policy(min_calibration_games=30,min_probability_bucket_games=20)
    legs,tickets,summary=run(sample(),policy)
    newest=legs.loc[legs.season.eq(2026)]
    assert newest.calibration_samples.eq(40).all()
    assert len(tickets)==1
    assert tickets.iloc[0].status=="RESEARCH_QUALIFIED"
    assert tickets.iloc[0].true_unit_return==-1.0
    assert summary.iloc[0].tickets==1


def test_no_trained_history_means_no_tickets():
    policy=V7Policy(min_calibration_games=100,min_probability_bucket_games=35)
    _,tickets,summary=run(sample(),policy)
    assert tickets.empty
    assert summary.empty
