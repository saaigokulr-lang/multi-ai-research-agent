"""create initial tables

Revision ID: 9af908d04c01
Revises:
Create Date: 2026-09-19 16:23:35.723855

Written by hand rather than via ``--autogenerate``: no live Postgres/Supabase
instance was reachable from this environment (DATABASE_URL in .env is still
a local placeholder), and autogenerate needs a real connection to diff
against. The tables below mirror app/database/models.py exactly.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9af908d04c01'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "research_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("sub_questions", sa.JSON(), nullable=True),
        sa.Column("research_iterations", sa.Integer(), nullable=False),
        sa.Column("research_incomplete", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_runs_status", "research_runs", ["status"], unique=False
    )

    op.create_table(
        "findings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("sub_question", sa.Text(), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_findings_run_id", "findings", ["run_id"], unique=False)

    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index("ix_reports_run_id", "reports", ["run_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_reports_run_id", table_name="reports")
    op.drop_table("reports")
    op.drop_index("ix_findings_run_id", table_name="findings")
    op.drop_table("findings")
    op.drop_index("ix_research_runs_status", table_name="research_runs")
    op.drop_table("research_runs")
