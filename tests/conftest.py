"""Shared fixtures. The API reads DATABASE_URL and ADMIN_SECRET_KEY at import
time, so they are set here, before any test module imports api.*."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# TEST_DATABASE_URL (e.g. the Postgres service in CI) wins; otherwise a fresh SQLite file.
if os.environ.get("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
else:
    _DB_DIR = tempfile.mkdtemp(prefix="promptguard-tests-")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{Path(_DB_DIR, 'test.db').as_posix()}"
os.environ["ADMIN_SECRET_KEY"] = "test-admin-secret"
os.environ.pop("ENABLE_HEAVY_MODELS", None)
os.environ.pop("CORS_ALLOW_ORIGINS", None)

ADMIN_HEADERS = {"X-Admin-Key": "test-admin-secret"}


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def api_key(client) -> str:
    response = client.post("/admin/api-keys", json={"org_name": "pytest"}, headers=ADMIN_HEADERS)
    assert response.status_code == 200
    return response.json()["key"]


@pytest.fixture()
def auth_headers(api_key) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture(scope="session")
def test_session():
    """Session factory for direct DB checks from test code.

    Tests run these with asyncio.run, i.e. on a new event loop each time. asyncpg
    connections are bound to the loop that created them, so the app's pooled
    engine (used on the TestClient's loop) cannot be shared; NullPool opens a
    fresh connection per session instead.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from api.database import DATABASE_URL, _engine_kwargs

    kwargs = {k: v for k, v in _engine_kwargs.items() if k not in {"pool_pre_ping", "pool_recycle"}}
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool, **kwargs)
    return async_sessionmaker(engine, expire_on_commit=False)
