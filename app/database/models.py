"""SQLAlchemy ORM models for research run persistence."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResearchRun(Base):
    """A single research run, tracked from submission through completion."""

    __tablename__ = "research_runs"
    __table_args__ = (Index("ix_research_runs_status", "status"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    sub_questions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    research_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    research_incomplete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    budget_exceeded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class Finding(Base):
    """A single sourced claim extracted during a research run."""

    __tablename__ = "findings"
    __table_args__ = (Index("ix_findings_run_id", "run_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("research_runs.id"), nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    sub_question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Report(Base):
    """The final Markdown report produced for a research run."""

    __tablename__ = "reports"
    __table_args__ = (Index("ix_reports_run_id", "run_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("research_runs.id"), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ExecutionTrace(Base):
    """One agent node's execution record (tokens, latency, outcome) within a run."""

    __tablename__ = "execution_traces"
    __table_args__ = (Index("ix_execution_traces_run_id", "run_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("research_runs.id"), nullable=False)
    node_name: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
