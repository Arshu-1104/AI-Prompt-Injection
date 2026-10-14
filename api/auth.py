from __future__ import annotations
import hashlib, hmac, os, secrets
from pathlib import Path
import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import ApiKey, get_db

ADMIN_SECRET = os.environ.get("ADMIN_SECRET_KEY", "")

# Leading characters of a raw key stored in plain text so a key can be found with
# one indexed query and a single bcrypt check, instead of bcrypt against every key.
# The remaining ~31 characters (~186 bits) stay secret.
KEY_PREFIX_LENGTH = 12


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


def key_prefix(raw: str) -> str:
    return raw[:KEY_PREFIX_LENGTH]

def hash_key(raw: str) -> str:
    return bcrypt.hashpw(raw.encode(), bcrypt.gensalt(rounds=12)).decode()

def verify_key(raw: str, hashed: str) -> bool:
    return bcrypt.checkpw(raw.encode(), hashed.encode())

def hash_input(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

async def get_current_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    # Dashboard requests use the server-only admin credential; external clients
    # must continue to authenticate with a bearer API key.
    admin_key = request.headers.get("X-Admin-Key", "")
    if ADMIN_SECRET and hmac.compare_digest(admin_key, ADMIN_SECRET):
        result = await db.execute(select(ApiKey).where(ApiKey.org_name == "__dashboard__"))
        dashboard_key = result.scalar_one_or_none()
        if dashboard_key:
            return dashboard_key
        raw_dashboard_key = generate_api_key()
        dashboard_key = ApiKey(
            key_hash=hash_key(raw_dashboard_key),
            key_prefix=key_prefix(raw_dashboard_key),
            org_name="__dashboard__",
            rate_limit_per_minute=600,
        )
        db.add(dashboard_key)
        await db.commit()
        await db.refresh(dashboard_key)
        return dashboard_key

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    raw = auth.removeprefix("Bearer ").strip()
    if len(raw) <= KEY_PREFIX_LENGTH:
        raise HTTPException(status_code=401, detail="Invalid API key")
    result = await db.execute(
        select(ApiKey).where(ApiKey.is_active == True, ApiKey.key_prefix == key_prefix(raw))
    )
    for row in result.scalars():
        if verify_key(raw, row.key_hash):
            request.state.api_key_id = row.id
            return row

    # Keys created before key_prefix existed: scan only those, and backfill the
    # prefix on a match so the next request takes the indexed path.
    legacy = await db.execute(select(ApiKey).where(ApiKey.is_active == True, ApiKey.key_prefix.is_(None)))
    for row in legacy.scalars():
        if verify_key(raw, row.key_hash):
            row.key_prefix = key_prefix(raw)
            await db.commit()
            request.state.api_key_id = row.id
            return row
    raise HTTPException(status_code=401, detail="Invalid API key")

def require_admin(request: Request) -> None:
    provided = request.headers.get("X-Admin-Key", "")
    if not ADMIN_SECRET or not hmac.compare_digest(provided, ADMIN_SECRET):
        raise HTTPException(status_code=403, detail="Admin access required")
