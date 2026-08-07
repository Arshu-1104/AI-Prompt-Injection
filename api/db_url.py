"""Normalizes a DATABASE_URL for use with SQLAlchemy's asyncpg driver.

Managed Postgres providers hand back libpq-style connection strings
(e.g. "postgresql://...?sslmode=require&channel_binding=require"), but
asyncpg's connect() doesn't accept "sslmode" or "channel_binding" as
keyword arguments — it raises TypeError at connection time. This
function strips those query params and returns the equivalent
connect_args so SSL is still enforced where the provider requires it.
"""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# Query params libpq/psycopg understand but asyncpg's connect() does not.
_ASYNCPG_INCOMPATIBLE_PARAMS = {"sslmode", "channel_binding"}


def normalize_async_db_url(raw_url: str) -> tuple[str, dict]:
    """Return (url, connect_args) safe to pass to create_async_engine."""
    url = raw_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    parts = urlsplit(url)
    query_pairs = parse_qsl(parts.query, keep_blank_values=True)

    connect_args: dict = {}
    kept_pairs = []
    require_ssl = False
    for key, value in query_pairs:
        if key in _ASYNCPG_INCOMPATIBLE_PARAMS:
            if key == "sslmode" and value in {"require", "verify-ca", "verify-full"}:
                require_ssl = True
            continue
        kept_pairs.append((key, value))

    if require_ssl:
        connect_args["ssl"] = "require"

    new_query = urlencode(kept_pairs)
    normalized_url = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, new_query, parts.fragment)
    )
    return normalized_url, connect_args