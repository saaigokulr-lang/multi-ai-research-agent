"""Shared schemas for the research workflow."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Structured output produced by the planner agent."""

    sub_questions: list[str]


class Finding(BaseModel):
    """A single claim extracted from a search result, with its source."""

    claim: str
    source_url: str
    snippet: str
    sub_question: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


MAX_RESEARCH_ITERATIONS: int = 2


class ResearchState(BaseModel):
    """State threaded through the research graph."""

    question: str
    sub_questions: list[str] = []
    findings: list[Finding] = []
    research_complete: bool = False
    research_iterations: int = 0
    research_incomplete: bool = False
    final_report: str = ""
