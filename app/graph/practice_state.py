"""State schema for the practice LangGraph graph."""

from pydantic import BaseModel


class PracticeState(BaseModel):
    """Minimal state passed between nodes in the practice graph."""

    name: str
    message: str = ""
