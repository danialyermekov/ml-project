from pathlib import Path

import joblib
import pandas as pd

from ml_project.config import ARTIFACTS, TEST


def predict(
    test_path: Path = TEST,
    artifacts_dir: Path = ARTIFACTS,
    model_path: Path = ARTIFACTS / "forest.joblib",
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(test_path, header=None)

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    preds_df = pd.DataFrame({"Predictions": preds})
    preds_df.to_csv(artifacts_dir / "predictions.csv", index=False)

    print("Preds are ready")


if __name__ == "__main__":
    predict()
