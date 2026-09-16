from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestClassifier

from ml_project.config import ARTIFACTS
from ml_project.schema import PredictionRequest, PredictionResponse

models: dict[str, RandomForestClassifier] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    models["forest"] = joblib.load(ARTIFACTS / "forest.joblib")

    yield

    models.clear()


app = FastAPI(title="ML System", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    model = models["forest"]
    expected_features = model.n_features_in_

    if len(request.features) != expected_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {expected_features} features,got {len(request.features)}"
            ),
        )

    prediction = int(model.predict([request.features])[0])

    return PredictionResponse(prediction=prediction)
