"""
Sentiment Analyzer — FastAPI Backend
Compares BERT, TextBlob, and VADER side-by-side.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from .models import BERTAnalyzer, TextBlobAnalyzer, VADERAnalyzer

app = FastAPI(
    title="Sentiment Analyzer API",
    description="Compare NLP models: DistilBERT vs TextBlob vs VADER",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────────────────────
bert_model    = BERTAnalyzer()
textblob_model = TextBlobAnalyzer()
vader_model   = VADERAnalyzer()


# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, example="This movie was absolutely fantastic!")
    models: list[str] = Field(default=["bert", "textblob", "vader"])

class ModelResult(BaseModel):
    model_name: str
    label: str          # POSITIVE / NEGATIVE / NEUTRAL
    confidence: float   # 0.0 – 1.0
    scores: dict        # raw per-label scores
    latency_ms: float

class AnalyzeResponse(BaseModel):
    text: str
    results: list[ModelResult]
    consensus: str      # majority vote label
    processing_time_ms: float


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., max_length=100)

class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "models_loaded": ["bert", "textblob", "vader"]}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    total_start = time.perf_counter()
    results = []

    model_map = {
        "bert":      bert_model,
        "textblob":  textblob_model,
        "vader":     vader_model,
    }

    for name in req.models:
        if name not in model_map:
            continue
        t0 = time.perf_counter()
        result = model_map[name].analyze(req.text)
        latency = (time.perf_counter() - t0) * 1000
        results.append(ModelResult(
            model_name=name,
            label=result["label"],
            confidence=result["confidence"],
            scores=result["scores"],
            latency_ms=round(latency, 2),
        ))

    # Consensus: majority vote
    labels = [r.label for r in results]
    consensus = max(set(labels), key=labels.count)
    total_ms = (time.perf_counter() - total_start) * 1000

    return AnalyzeResponse(
        text=req.text,
        results=results,
        consensus=consensus,
        processing_time_ms=round(total_ms, 2),
    )


@app.post("/batch")
def batch_analyze(req: BatchRequest):
    if len(req.texts) > 100:
        raise HTTPException(status_code=422, detail="Max 100 texts per batch")
    return [
        analyze(AnalyzeRequest(text=t))
        for t in req.texts
        if t.strip()
    ]


@app.get("/")
def root():
    return {"message": "Sentiment Analyzer API", "docs": "/docs", "health": "/health"}
