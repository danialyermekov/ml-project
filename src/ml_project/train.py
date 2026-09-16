import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ml_project.config import ARTIFACTS, RANDOM_STATE, TRAIN, TRAIN_LABELS


def train(
    train_path: Path = TRAIN,
    labels_path: Path = TRAIN_LABELS,
    artifacts_dir: Path = ARTIFACTS,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(train_path, header=None)
    y = pd.read_csv(labels_path, header=None).squeeze("columns")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, stratify=y, shuffle=True, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    metrics = {
        "accuracy": accuracy_score(y_valid, preds),
        "recall": recall_score(y_valid, preds),
        "precision": precision_score(y_valid, preds),
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(model, artifacts_dir / "forest.joblib")


if __name__ == "__main__":
    train()
