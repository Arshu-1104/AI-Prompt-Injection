"""api_key_prefix

Adds api_keys.key_prefix so bearer auth looks a key up by an indexed prefix and
runs one bcrypt check, instead of bcrypt-verifying every active key. Existing
keys keep working: they are matched by a fallback scan and backfilled on use.

Revision ID: 0002_api_key_prefix
Revises: 0001_initial_schema
Create Date: 2026-10-13 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_api_key_prefix"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("key_prefix", sa.String(length=16), nullable=True))
    op.create_index(op.f("ix_api_keys_key_prefix"), "api_keys", ["key_prefix"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_api_keys_key_prefix"), table_name="api_keys")
    op.drop_column("api_keys", "key_prefix")
