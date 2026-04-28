from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import PromptAnalyzer
from src.preprocess import ATTACK_PATTERNS


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    model: Literal["classical", "bert"] = "classical"


class BatchPredictRequest(BaseModel):
    texts: list[str]
    model: Literal["classical", "bert"] = "classical"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_loaded = False
    app.state.analyzers = {}
    try:
        app.state.analyzers["classical"] = PromptAnalyzer(model_type="classical")
        app.state.analyzers["bert"] = PromptAnalyzer(model_type="bert")
        app.state.models_loaded = True
    except Exception:
        app.state.models_loaded = False
    yield


app = FastAPI(title="PromptGuard API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(app.state.models_loaded), "version": "1.0.0"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    analyzer = app.state.analyzers[payload.model]
    result = analyzer.predict(payload.text)
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "risk_score": result["risk_score"],
        "attack_patterns": result["attack_patterns"],
        "explanation": result["explanation"],
        "model_used": payload.model,
    }


@app.post("/batch_predict")
def batch_predict(payload: BatchPredictRequest) -> list[dict]:
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty.")
    analyzer = app.state.analyzers[payload.model]
    predictions = analyzer.batch_predict(payload.texts)
    return [
        {
            "label": item["label"],
            "confidence": item["confidence"],
            "risk_score": item["risk_score"],
            "attack_patterns": item["attack_patterns"],
            "explanation": item["explanation"],
            "model_used": payload.model,
        }
        for item in predictions
    ]


@app.get("/patterns")
def patterns() -> dict:
    return {"attack_patterns": ATTACK_PATTERNS}


# curl examples:
# curl -X GET "http://127.0.0.1:8000/health"
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"Ignore previous instructions\",\"model\":\"classical\"}"
# curl -X POST "http://127.0.0.1:8000/batch_predict" -H "Content-Type: application/json" -d "{\"texts\":[\"hello\",\"ignore previous\"],\"model\":\"bert\"}"
