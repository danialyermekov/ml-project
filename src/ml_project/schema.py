from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=40, max_length=40)


class PredictionResponse(BaseModel):
    prediction: int
