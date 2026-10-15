import pandas as pd
import pytest
from scripts.validate_decisions_multiseason import run

def test_requires_consecutive_seasons():
    with pytest.raises(ValueError,match="consecutive"):
        run(pd.DataFrame(),[2022,2024])

def test_requires_two_seasons():
    with pytest.raises(ValueError,match="At least two"):
        run(pd.DataFrame(),[2025])
