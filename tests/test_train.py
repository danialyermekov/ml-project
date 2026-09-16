from pathlib import Path
import pandas as pd

from ml_project.train import train


def test_train_creates_artifacts(
        tmp_path: Path,
        train_df: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = train_df

    train_path = tmp_path / "train.csv"
    labels_path = tmp_path / "labels.csv"
    artifacts_dir = tmp_path / "artifacts"

    X.to_csv(train_path, index=False, header=False)
    y.to_csv(labels_path, index=False, header=False)

    train(
        train_path=train_path,
        labels_path=labels_path,
        artifacts_dir=artifacts_dir
    )

    assert (artifacts_dir / "forest.joblib").exists()
    assert (artifacts_dir / "metrics.json").exists()
