"""Unit tests for research-loop routing. Pure state logic, no mocking needed."""

from app.graph.routing import route_after_supervisor
from app.schemas.research import ResearchState


def test_routes_to_researcher_when_research_incomplete() -> None:
    state = ResearchState(question="How is AI changing healthcare?", research_complete=False)
    assert route_after_supervisor(state) == "researcher"


def test_routes_to_writer_when_research_complete() -> None:
    state = ResearchState(question="How is AI changing healthcare?", research_complete=True)
    assert route_after_supervisor(state) == "writer"
