"""Shared schemas for the research workflow."""

from pydantic import BaseModel


class ResearchPlan(BaseModel):
    """Structured output produced by the planner agent."""

    sub_questions: list[str]


class ResearchState(BaseModel):
    """State threaded through the research graph."""

    question: str
    sub_questions: list[str] = []
    findings: list[dict] = []
    research_complete: bool = False
    final_report: str = ""
