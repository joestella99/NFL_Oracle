import pandas as pd
import pytest
from nfl_oracle.personnel import validate_personnel_snapshots,personnel_features

def data():
    return pd.DataFrame([dict(game_id='G',team='A',position_group='QB',status='out',
        observed_at='2025-09-07T13:00:00Z',kickoff='2025-09-07T17:00:00Z',
        source_url='https://example.org/nfl-official-report')])

def test_valid_snapshot():
    assert validate_personnel_snapshots(data()).iloc[0]['status']=='out'

def test_reject_post_kickoff():
    d=data();d['observed_at']='2025-09-07T18:00:00Z'
    with pytest.raises(ValueError,match='Lookahead'):validate_personnel_snapshots(d)

def test_missing_is_not_assumed_healthy():
    g=pd.DataFrame([dict(game_id='G',home_team='A',away_team='B')])
    p=personnel_features(g,data())
    assert p.loc[0,'home_qb_out']==1
    assert pd.isna(p.loc[0,'away_qb_out'])
    assert pd.isna(p.loc[0,'qb_out_diff'])

def test_invalid_source():
    d=data();d['source_url']='unknown'
    with pytest.raises(ValueError,match='source'):validate_personnel_snapshots(d)
