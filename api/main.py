from __future__ import annotations

import asyncio
import json
import logging
import os
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

from api.auth import generate_api_key, get_current_api_key, hash_input, hash_key, key_prefix, require_admin
from api.database import ApiKey, AsyncSessionLocal, Base, Prediction, engine, get_db, utcnow_naive
from src.model_sources import ModelUnavailableError
from src.predict import PromptAnalyzer
from src.preprocess import ATTACK_PATTERNS


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    model: Literal["classical", "bert", "guard"] = "classical"


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    model: Literal["classical", "bert", "guard"] = "classical"


class CreateKeyRequest(BaseModel):
    org_name: str = Field(min_length=1, max_length=200)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)


class SettingsPayload(BaseModel):
    store_raw_text: bool = False
    webhook_url: str | None = None
    risk_threshold: int = Field(default=85, ge=0, le=100)


# BUG 3 FIX: Added comment documenting known limitation.
# TODO: Move to Redis or PostgreSQL rate_limits table for persistence across restarts.
# Current implementation resets counters on every server restart, allowing a brief
# burst above the configured limit after each deploy.
_RATE_WINDOW: dict[str, deque[float]] = defaultdict(deque)

# uvicorn configures this logger, so INFO lines show up in the server output.
logger = logging.getLogger("uvicorn.error")
MODEL_TYPES = ("classical", "bert", "guard")

# The event loop keeps only weak references to tasks, so a fire-and-forget
# prediction-log task can be garbage-collected before it finishes. Hold strong
# references until each task completes, and surface failures in the log.
_background_tasks: set[asyncio.Task] = set()
SHUTDOWN_DRAIN_SECONDS = 5.0


def _on_background_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error("Background task failed", exc_info=task.exception())


def _spawn_background(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_on_background_task_done)
    return task


def _heavy_models_enabled() -> bool:
    return os.environ.get("ENABLE_HEAVY_MODELS", "false").lower() == "true"


# Public reason codes for an unavailable model (GET /health, 503 responses).
# Details such as file paths and exception text are logged server-side only.
REASON_HEAVY_DISABLED = "heavy_disabled"


def _load_analyzer(app: FastAPI, model_type: str, **kwargs) -> None:
    t0 = time.monotonic()
    try:
        analyzer = PromptAnalyzer(model_type=model_type, **kwargs)
    except ModelUnavailableError as exc:
        app.state.model_status[model_type] = {"available": False, "reason": exc.code}
        logger.warning("Model '%s' unavailable (%s): %s", model_type, exc.code, exc)
        return
    except Exception:
        app.state.model_status[model_type] = {"available": False, "reason": "load_failed"}
        logger.exception("Model '%s' failed to load (load_failed)", model_type)
        return
    app.state.analyzers[model_type] = analyzer
    app.state.model_status[model_type] = {"available": True, "reason": None}
    logger.info("Model '%s' loaded from %s in %.1fs", model_type, analyzer.source, time.monotonic() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    app.state.analyzers = {}
    app.state.model_status = {}
    # Heavy transformer models are opt-in: they need torch/transformers (requirements-heavy.txt)
    # and roughly 1.5-3 GB RAM, more than small hosting tiers provide.
    if _heavy_models_enabled():
        _load_analyzer(app, "bert")
        _load_analyzer(app, "guard")
    else:
        for model_type in ("bert", "guard"):
            app.state.model_status[model_type] = {"available": False, "reason": REASON_HEAVY_DISABLED}
        logger.info("ENABLE_HEAVY_MODELS is not 'true'; skipping bert and guard.")
    # Classical escalates uncertain SUSPICIOUS results to guard, but only when guard is
    # already loaded (reusing it), so ENABLE_HEAVY_MODELS=false never loads ModernBERT.
    guard = app.state.analyzers.get("guard")
    _load_analyzer(app, "classical", escalate_on_uncertain=guard is not None, escalation_analyzer=guard)
    app.state.models_loaded = len(app.state.analyzers) > 0
    logger.info(
        "Models available: %s; unavailable: %s",
        sorted(app.state.analyzers) or "none",
        sorted(set(MODEL_TYPES) - set(app.state.analyzers)) or "none",
    )
    yield
    # Let in-flight prediction logs reach the database before shutting down.
    if _background_tasks:
        _, pending = await asyncio.wait(set(_background_tasks), timeout=SHUTDOWN_DRAIN_SECONDS)
        if pending:
            logger.warning("Shutdown: %d prediction log writes did not finish", len(pending))


app = FastAPI(title="PromptGuard API", version="1.0.0", lifespan=lifespan)


def _key_func(request: Request) -> str:
    return str(getattr(request.state, "api_key_id", "anon"))


limiter = Limiter(key_func=_key_func)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: PlainTextResponse("Rate limit exceeded", status_code=429))
app.add_middleware(SlowAPIMiddleware)
# Browser origins allowed to call the API directly, comma-separated (e.g.
# "https://app.example.com"); "*" allows any. Default: none. First-party clients
# don't need CORS: the dashboard calls the API server-side through /api/proxy,
# and the extension calls it from its background worker under host_permissions.
CORS_ALLOW_ORIGINS = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    # The API is bearer-token based, so credentialed CORS is unnecessary (and
    # invalid with a wildcard origin).
    allow_credentials=False,
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
    return {
        "status": "ok",
        "models_loaded": bool(app.state.models_loaded),
        "heavy_models_enabled": _heavy_models_enabled(),
        "models": {
            model_type: {
                "available": model_type in app.state.analyzers,
                "reason": app.state.model_status.get(model_type, {}).get("reason"),
            }
            for model_type in MODEL_TYPES
        },
        "version": "1.0.0",
    }


def _get_analyzer(model_type: str) -> PromptAnalyzer:
    analyzer = app.state.analyzers.get(model_type)
    if analyzer is None:
        reason = app.state.model_status.get(model_type, {}).get("reason") or "load_failed"
        raise HTTPException(status_code=503, detail=f"Model '{model_type}' is unavailable ({reason}).")
    return analyzer


@app.post("/predict")
async def predict(
    payload: PredictRequest,
    request: Request,
    api_key: ApiKey = Depends(get_current_api_key),
) -> dict:
    request.state.api_key_id = api_key.id
    await _enforce_rate_limit(api_key)
    t0 = time.monotonic()
    analyzer = _get_analyzer(payload.model)
    result = await run_in_threadpool(analyzer.predict, payload.text)
    latency_ms = (time.monotonic() - t0) * 1000
    _spawn_background(_log_prediction(result, payload, api_key, latency_ms))
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
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty.")
    analyzer = _get_analyzer(payload.model)
    t0 = time.monotonic()
    predictions = await run_in_threadpool(analyzer.batch_predict, payload.texts)
    total_ms = (time.monotonic() - t0) * 1000
    # BUG 1 FIX: Each item is now logged with its own per-item latency instead of
    # the total batch latency. This prevents /metrics latency stats from being
    # inflated when batch requests are common.
    per_item_ms = total_ms / max(len(payload.texts), 1)
    for text, item in zip(payload.texts, predictions):
        _spawn_background(
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
        key_prefix=key_prefix(raw),
        org_name=payload.org_name,
        rate_limit_per_minute=payload.rate_limit_per_minute,
    )
    db.add(key)
    await db.commit()
    await db.refresh(key)
    return {"id": str(key.id), "key": raw, "warning": "Store this key now; it will not be shown again."}


@app.get("/admin/api-keys", dependencies=[Depends(require_admin)])
async def list_api_keys(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    return [
        {
            "id": str(key.id),
            "org_name": key.org_name,
            "key_prefix": key.key_prefix,
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
    start = utcnow_naive() - timedelta(days=days)

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
        day = (utcnow_naive() - timedelta(days=days - i - 1)).date().isoformat()
        daily_map[day] = {"date": day, "SAFE": 0, "SUSPICIOUS": 0, "MALICIOUS": 0}

    # Merge database counts into the pre-filled map
    for row in daily_rows:
        # Postgres returns a date, SQLite's date() a 'YYYY-MM-DD' string.
        day = row.date if isinstance(row.date, str) else row.date.isoformat()
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