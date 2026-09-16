import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from collections.abc import Iterator
import ml_project.api as api_module
from ml_project.config import RANDOM_STATE

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


@pytest.fixture
def api_model() -> RandomForestClassifier:
    X = np.vstack(
        [
            np.zeros((10, 40)),
            np.ones((10, 40))
        ]
    )
    y = np.array([0] * 10 + [1] * 10)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)

    return model


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    api_model: RandomForestClassifier
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        api_module.joblib,
        "load",
        lambda path: api_model
    )

    with TestClient(api_module.app) as test_client:
        yield test_client
