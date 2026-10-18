"""A database whose tables predate migration 0002 (no api_keys.key_prefix) and that
has no alembic_version row, because the app's create_all() built it. That is the
state that made production return 500 for every route that reads api_keys."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from tests.conftest import ROOT

PRE_0002_SCHEMA = """
CREATE TABLE api_keys (id VARCHAR(36) PRIMARY KEY, key_hash VARCHAR(100) NOT NULL UNIQUE,
  org_name VARCHAR(200) NOT NULL, created_at DATETIME, rate_limit_per_minute INTEGER,
  is_active BOOLEAN, store_raw_text BOOLEAN);
CREATE TABLE predictions (id VARCHAR(36) PRIMARY KEY, timestamp DATETIME,
  api_key_id VARCHAR(36) REFERENCES api_keys(id), input_hash VARCHAR(64) NOT NULL, input_raw VARCHAR,
  label VARCHAR(20) NOT NULL, confidence FLOAT NOT NULL, risk_score FLOAT NOT NULL,
  model_used VARCHAR(50) NOT NULL, attack_patterns JSON, latency_ms FLOAT NOT NULL);
CREATE INDEX ix_predictions_timestamp ON predictions (timestamp);
INSERT INTO api_keys VALUES ('00000000-0000-0000-0000-000000000001', 'legacy-hash', 'legacy-org',
  '2026-06-20 00:00:00', 60, 1, 0);
"""


def _pre_0002_db(tmp_path: Path) -> Path:
    db = tmp_path / "drift.db"
    with sqlite3.connect(db) as con:
        con.executescript(PRE_0002_SCHEMA)
    return db


def _upgrade_head(db: Path, monkeypatch) -> None:
    # No alembic.ini: its logging config would disable the app's loggers for later tests.
    config = Config()
    config.set_main_option("script_location", str(ROOT / "api" / "migrations"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db.as_posix()}")
    command.upgrade(config, "head")


def _columns(db: Path, table: str) -> set[str]:
    with sqlite3.connect(db) as con:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def _indexes(db: Path, table: str) -> set[str]:
    with sqlite3.connect(db) as con:
        return {row[1] for row in con.execute(f"PRAGMA index_list({table})")}


def _version(db: Path) -> str:
    with sqlite3.connect(db) as con:
        return con.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def test_missing_columns_reports_the_pre_0002_table(tmp_path):
    from api.database import Base, missing_columns

    engine = create_engine(f"sqlite:///{_pre_0002_db(tmp_path).as_posix()}")
    with engine.connect() as connection:
        assert missing_columns(connection) == ["api_keys.key_prefix"]
    fresh = create_engine(f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
    Base.metadata.create_all(fresh)
    with fresh.connect() as connection:
        assert missing_columns(connection) == []


def test_upgrade_head_adopts_an_unversioned_pre_0002_database(tmp_path, monkeypatch):
    db = _pre_0002_db(tmp_path)
    _upgrade_head(db, monkeypatch)
    assert "key_prefix" in _columns(db, "api_keys")
    assert "ix_api_keys_key_prefix" in _indexes(db, "api_keys")
    assert _version(db) == "0002_api_key_prefix"
    with sqlite3.connect(db) as con:  # existing keys survive; they get a prefix on first use
        assert con.execute("SELECT org_name, key_prefix FROM api_keys").fetchall() == [("legacy-org", None)]
    _upgrade_head(db, monkeypatch)  # running it again is a no-op
    assert _version(db) == "0002_api_key_prefix"


def test_upgrade_head_adopts_a_database_built_by_current_create_all(tmp_path, monkeypatch):
    from api.database import Base

    db = tmp_path / "current.db"
    Base.metadata.create_all(create_engine(f"sqlite:///{db.as_posix()}"))
    _upgrade_head(db, monkeypatch)
    assert _version(db) == "0002_api_key_prefix"


def test_upgrade_head_on_an_empty_database(tmp_path, monkeypatch):
    from api.database import missing_columns

    db = tmp_path / "empty.db"
    _upgrade_head(db, monkeypatch)
    with create_engine(f"sqlite:///{db.as_posix()}").connect() as connection:
        assert missing_columns(connection) == []
    assert "ix_predictions_timestamp" in _indexes(db, "predictions")


def test_startup_logs_missing_columns(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    import api.main as main

    monkeypatch.setattr(main, "missing_columns", lambda connection: ["api_keys.key_prefix"])
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        with TestClient(main.app):
            pass
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert any("api_keys.key_prefix" in m and "alembic upgrade head" in m for m in messages), messages
