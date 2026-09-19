"""Request/response schemas for the research API."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateResearchRequest(BaseModel):
    """Body for POST /research."""

    question: str = Field(min_length=5)
    idempotency_key: str | None = None


class CreateResearchResponse(BaseModel):
    """Response for POST /research."""

    run_id: str
    status: str


class ResearchStatusResponse(BaseModel):
    """Response for GET /research/{run_id}."""

    run_id: str
    status: str
    question: str
    created_at: datetime
    sub_questions: list[str] | None = None
    research_iterations: int | None = None
    research_incomplete: bool | None = None
    final_report: str | None = None
    error: str | None = None


class TraceStep(BaseModel):
    """One agent node's execution record within a research run."""

    node_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str
    started_at: datetime
    completed_at: datetime
    error: str | None = None


class TraceResponse(BaseModel):
    """Response for GET /research/{run_id}/trace."""

    run_id: str
    status: str
    steps: list[TraceStep]
    total_tokens_used: int
    budget_exceeded: bool
