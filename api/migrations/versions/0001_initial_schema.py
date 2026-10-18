"""initial_schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-20 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A database first built by the app's create_all() has these tables but no
    # alembic_version row. Skip whatever exists so `alembic upgrade head` adopts
    # it and runs the later migrations, instead of failing with "already exists".
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("api_keys"):
        _create_api_keys()
    if not inspector.has_table("predictions"):
        _create_predictions()
    if "ix_predictions_timestamp" not in {index["name"] for index in inspector.get_indexes("predictions")}:
        op.create_index(op.f("ix_predictions_timestamp"), "predictions", ["timestamp"], unique=False)


def _create_api_keys() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(length=100), nullable=False),
        sa.Column("org_name", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("rate_limit_per_minute", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("store_raw_text", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
    )


def _create_predictions() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("input_raw", sa.String(), nullable=True),
        sa.Column("label", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("model_used", sa.String(length=50), nullable=False),
        sa.Column("attack_patterns", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_timestamp"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("api_keys")
