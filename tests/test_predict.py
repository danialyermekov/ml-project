from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml_project.predict import predict


def test_predict_creates_predictions(
    tmp_path: Path, train_df: tuple[pd.DataFrame, pd.Series]
) -> None:
    test_path = tmp_path / "test.csv"
    artifacts_dir = tmp_path / "artifacts"
    model_path = artifacts_dir / "model.joblib"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train_df

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    X_test = pd.DataFrame(
        {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    X_test.to_csv(test_path, index=False, header=False)

    predict(test_path=test_path, artifacts_dir=artifacts_dir, model_path=model_path)
    prediction_path = artifacts_dir / "predictions.csv"
    assert prediction_path.exists()

    predictions = pd.read_csv(prediction_path)
    assert len(predictions) == len(X_test)

    assert "Predictions" in predictions
