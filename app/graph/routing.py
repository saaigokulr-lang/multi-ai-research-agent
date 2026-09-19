"""Conditional routing logic for the research loop.

Pure Python state checks, no AI judgment: by the time this runs,
``supervisor_node`` has already applied the iteration-limit safety cap, so
``state.research_complete`` is the single source of truth for where to go
next.
"""

from app.schemas.research import ResearchState


def route_after_supervisor(state: ResearchState) -> str:
    """Return the next node name based on the supervisor's decision."""
    if state.research_complete:
        return "writer"
    return "researcher"
