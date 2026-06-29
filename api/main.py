from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from starlette.responses import PlainTextResponse

from api.auth import generate_api_key, get_current_api_key, hash_input, hash_key, require_admin
from api.database import ApiKey, AsyncSessionLocal, Prediction, get_db
from src.predict import PromptAnalyzer
from src.preprocess import ATTACK_PATTERNS


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    model: Literal["classical", "bert", "guard"] = "classical"


class BatchPredictRequest(BaseModel):
    texts: list[str]
    model: Literal["classical", "bert", "guard"] = "classical"


class CreateKeyRequest(BaseModel):
    org_name: str
    rate_limit_per_minute: int = 60


class SettingsPayload(BaseModel):
    store_raw_text: bool = False
    webhook_url: str | None = None
    risk_threshold: int = Field(default=85, ge=0, le=100)


# BUG 3 FIX: Added comment documenting known limitation.
# TODO: Move to Redis or PostgreSQL rate_limits table for persistence across restarts.
# Current implementation resets counters on every server restart, allowing a brief
# burst above the configured limit after each deploy.
_RATE_WINDOW: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create database tables (works for both SQLite and PostgreSQL)
    from api.database import Base, engine as db_engine
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.models_loaded = False
    app.state.analyzers = {}
    try:
        app.state.analyzers["classical"] = PromptAnalyzer(model_type="classical", escalate_on_uncertain=True)
    except Exception:
        pass
    try:
        app.state.analyzers["bert"] = PromptAnalyzer(model_type="bert")
    except Exception:
        pass
    try:
        app.state.analyzers["guard"] = PromptAnalyzer(model_type="guard")
    except Exception:
        pass
    app.state.models_loaded = len(app.state.analyzers) > 0
    yield


app = FastAPI(title="PromptGuard API", version="1.0.0", lifespan=lifespan)


def _key_func(request: Request) -> str:
    return str(getattr(request.state, "api_key_id", "anon"))


limiter = Limiter(key_func=_key_func)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: PlainTextResponse("Rate limit exceeded", status_code=429))
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _enforce_rate_limit(api_key: ApiKey) -> None:
    now = time.monotonic()
    bucket = _RATE_WINDOW[str(api_key.id)]
    while bucket and now - bucket[0] >= 60:
        bucket.popleft()
    if len(bucket) >= api_key.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


async def _log_prediction(result: dict, payload: PredictRequest, api_key: ApiKey, latency_ms: float) -> None:
    async with AsyncSessionLocal() as db:
        row = Prediction(
            api_key_id=api_key.id,
            input_hash=hash_input(payload.text),
            input_raw=payload.text if api_key.store_raw_text else None,
            label=result["label"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            model_used=payload.model,
            attack_patterns=result["attack_patterns"],
            latency_ms=latency_ms,
        )
        db.add(row)
        await db.commit()


def _settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "settings.json"


def _read_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {"store_raw_text": False, "webhook_url": None, "risk_threshold": 85}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_settings(settings: dict) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(app.state.models_loaded), "version": "1.0.0"}


@app.post("/predict")
async def predict(
    payload: PredictRequest,
    request: Request,
    api_key: ApiKey = Depends(get_current_api_key),
) -> dict:
    request.state.api_key_id = api_key.id
    await _enforce_rate_limit(api_key)
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    t0 = time.monotonic()
    analyzer = app.state.analyzers.get(payload.model)
    if not analyzer:
        raise HTTPException(status_code=503, detail=f"Model '{payload.model}' unavailable.")
    result = await run_in_threadpool(analyzer.predict, payload.text)
    latency_ms = (time.monotonic() - t0) * 1000
    asyncio.create_task(_log_prediction(result, payload, api_key, latency_ms))
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "risk_score": result["risk_score"],
        "attack_patterns": result["attack_patterns"],
        "explanation": result["explanation"],
        "token_highlights": result.get("token_highlights", {}),
        "escalated": result.get("escalated", False),
        "model_used": payload.model,
    }


@app.post("/batch_predict")
async def batch_predict(
    payload: BatchPredictRequest,
    request: Request,
    api_key: ApiKey = Depends(get_current_api_key),
) -> list[dict]:
    request.state.api_key_id = api_key.id
    await _enforce_rate_limit(api_key)
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty.")
    analyzer = app.state.analyzers.get(payload.model)
    if not analyzer:
        raise HTTPException(status_code=503, detail=f"Model '{payload.model}' unavailable.")
    t0 = time.monotonic()
    predictions = await run_in_threadpool(analyzer.batch_predict, payload.texts)
    total_ms = (time.monotonic() - t0) * 1000
    # BUG 1 FIX: Each item is now logged with its own per-item latency instead of
    # the total batch latency. This prevents /metrics latency stats from being
    # inflated when batch requests are common.
    per_item_ms = total_ms / max(len(payload.texts), 1)
    for text, item in zip(payload.texts, predictions):
        asyncio.create_task(
            _log_prediction(item, PredictRequest(text=text, model=payload.model), api_key, per_item_ms)
        )
    return [
        {
            "label": item["label"],
            "confidence": item["confidence"],
            "risk_score": item["risk_score"],
            "attack_patterns": item["attack_patterns"],
            "explanation": item["explanation"],
            "token_highlights": item.get("token_highlights", {}),
            "escalated": item.get("escalated", False),
            "model_used": payload.model,
        }
        for item in predictions
    ]


@app.get("/patterns")
def patterns() -> dict:
    return {"attack_patterns": ATTACK_PATTERNS}


@app.post("/admin/api-keys", dependencies=[Depends(require_admin)])
async def create_api_key(payload: CreateKeyRequest, db: AsyncSession = Depends(get_db)):
    raw = generate_api_key()
    key = ApiKey(
        key_hash=hash_key(raw),
        org_name=payload.org_name,
        rate_limit_per_minute=payload.rate_limit_per_minute,
    )
    db.add(key)
    await db.commit()
    return {"id": str(key.id), "key": raw, "warning": "Store this key now; it will not be shown again."}


@app.get("/admin/api-keys", dependencies=[Depends(require_admin)])
async def list_api_keys(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    return [
        {
            "id": str(key.id),
            "org_name": key.org_name,
            "created_at": key.created_at.isoformat(),
            "rate_limit_per_minute": key.rate_limit_per_minute,
            "is_active": key.is_active,
            "store_raw_text": key.store_raw_text,
        }
        for key in result.scalars()
    ]


@app.delete("/admin/api-keys/{key_id}", dependencies=[Depends(require_admin)])
async def revoke_key(key_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(404, "Key not found")
    key.is_active = False
    await db.commit()
    return {"revoked": key_id}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(db: AsyncSession = Depends(get_db)) -> str:
    result = await db.execute(select(Prediction.label, func.count()).group_by(Prediction.label))
    counts = {row[0]: row[1] for row in result}
    lines = []
    for label in ["SAFE", "SUSPICIOUS", "MALICIOUS"]:
        lines.append(f'promptguard_requests_total{{label="{label}"}} {counts.get(label, 0)}')
    return "\n".join(lines) + "\n"


@app.get("/api/stats")
async def stats(
    days: int = Query(default=7, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(get_current_api_key),
) -> dict:
    from api.database import _dialect
    start = datetime.utcnow() - timedelta(days=days)

    if _dialect == "postgresql":
        from sqlalchemy import cast, Date
        date_expr = cast(Prediction.timestamp, Date)
    else:
        from sqlalchemy import func as _func
        date_expr = _func.date(Prediction.timestamp)

    # BUG 2 FIX: Replaced Python-side full-table scan with a SQL GROUP BY query.
    daily_result = await db.execute(
        select(
            date_expr.label("date"),
            Prediction.label,
            func.count().label("count"),
        )
        .where(Prediction.timestamp >= start)
        .group_by(date_expr, Prediction.label)
    )
    daily_rows = daily_result.all()

    pattern_result = await db.execute(
        select(Prediction.attack_patterns).where(Prediction.timestamp >= start)
    )
    pattern_counts: Counter[str] = Counter()
    for (patterns,) in pattern_result:
        pattern_counts.update(patterns or [])

    # Pre-fill every day in the requested range with zeros
    daily_map: dict[str, dict[str, int | str]] = {}
    for i in range(days):
        day = (datetime.utcnow() - timedelta(days=days - i - 1)).date().isoformat()
        daily_map[day] = {"date": day, "SAFE": 0, "SUSPICIOUS": 0, "MALICIOUS": 0}

    # Merge database counts into the pre-filled map
    for row in daily_rows:
        day = row.date.isoformat()
        if day not in daily_map:
            daily_map[day] = {"date": day, "SAFE": 0, "SUSPICIOUS": 0, "MALICIOUS": 0}
        daily_map[day][row.label] = row.count

    return {
        "daily": list(daily_map.values()),
        "top_patterns": [
            {"pattern": pattern, "count": count}
            for pattern, count in pattern_counts.most_common(5)
        ],
    }


@app.get("/api/logs")
async def logs(
    page: int = Query(default=1, ge=1),
    label: str | None = None,
    model: str | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(get_current_api_key),
) -> dict:
    _ = api_key
    per_page = 20
    filters = []
    if label:
        filters.append(Prediction.label == label)
    if model:
        filters.append(Prediction.model_used == model)
    if from_date:
        filters.append(Prediction.timestamp >= from_date)
    if to_date:
        filters.append(Prediction.timestamp <= to_date)
    count_stmt = select(func.count()).select_from(Prediction)
    stmt = select(Prediction).order_by(Prediction.timestamp.desc()).offset((page - 1) * per_page).limit(per_page)
    if filters:
        count_stmt = count_stmt.where(*filters)
        stmt = stmt.where(*filters)
    total = (await db.execute(count_stmt)).scalar_one()
    result = await db.execute(stmt)
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "items": [
            {
                "id": str(row.id),
                "timestamp": row.timestamp.isoformat(),
                "label": row.label,
                "confidence": row.confidence,
                "risk_score": row.risk_score,
                "model_used": row.model_used,
                "attack_patterns": row.attack_patterns or [],
                "explanation": None,
            }
            for row in result.scalars()
        ],
    }


@app.get("/api/settings")
async def get_settings(api_key: ApiKey = Depends(get_current_api_key)) -> dict:
    _ = api_key
    return _read_settings()


@app.patch("/api/settings")
async def update_settings(
    payload: SettingsPayload,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(get_current_api_key),
) -> dict:
    settings = payload.model_dump()
    _write_settings(settings)
    api_key.store_raw_text = payload.store_raw_text
    await db.commit()
    return settings