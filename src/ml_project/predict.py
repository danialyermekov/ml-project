import joblib
import pandas as pd
from ml_project.config import ARTIFACTS, TEST


def predict() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(TEST, header=None)

    model = joblib.load(ARTIFACTS / "forest.joblib")
    preds = model.predict(X_test)

    preds_df = pd.DataFrame({"Predictions": preds})
    preds_df.to_csv(ARTIFACTS / "predictions.csv", index=False)

    print("Preds are ready")
if __name__ == "__main__":
    predict()
