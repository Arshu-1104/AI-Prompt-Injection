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
