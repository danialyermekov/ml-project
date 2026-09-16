import pytest
import pandas as pd

@pytest.fixture
def train_df() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                1: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            }
        )
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    return X, y
