from __future__ import annotations
import os, uuid
from datetime import datetime
from pathlib import Path
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Use PostgreSQL if DATABASE_URL is set, otherwise fall back to local SQLite
_DATABASE_URL = os.environ.get("DATABASE_URL")
if _DATABASE_URL:
    DATABASE_URL = _DATABASE_URL
    # asyncpg dialect for PostgreSQL
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    _engine_kwargs: dict = {"echo": False, "pool_pre_ping": True}
else:
    _db_path = Path(__file__).resolve().parents[1] / "artifacts" / "promptguard.db"
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = f"sqlite+aiosqlite:///{_db_path}"
    _engine_kwargs = {"echo": False, "connect_args": {"check_same_thread": False}}
    print(f"[database] No DATABASE_URL set — using SQLite at {_db_path}")

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Use native UUID for PostgreSQL, String fallback for SQLite
import sqlalchemy as _sa
_dialect = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql"

def _uuid_col(primary_key: bool = False, foreign_key: str | None = None):
    """Return a UUID column compatible with both PostgreSQL and SQLite."""
    if _dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID
        col_type = PG_UUID(as_uuid=True)
    else:
        col_type = _sa.String(36)
    if foreign_key:
        return mapped_column(col_type, _sa.ForeignKey(foreign_key), nullable=True)
    return mapped_column(col_type, primary_key=primary_key, default=lambda: str(uuid.uuid4()))


class Base(DeclarativeBase):
    pass

class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(_sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    org_name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    store_raw_text: Mapped[bool] = mapped_column(Boolean, default=False)

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(_sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    api_key_id: Mapped[str | None] = mapped_column(_sa.String(36), _sa.ForeignKey("api_keys.id"), nullable=True)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    attack_patterns: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
