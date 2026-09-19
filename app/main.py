"""FastAPI application entrypoint for the Multi-Agent Research Assistant.

Run with ``python -m app.main`` (or ``uvicorn app.main:app``) to serve the
/research API.
"""

import uvicorn
from fastapi import FastAPI

from app.api.routes import router as research_router
from app.core.config import get_settings
from app.graph.research_graph import build_research_graph

get_settings()

# Built once at import time and reused across requests -- a compiled
# LangGraph graph is safe to share since it holds no per-run state itself.
research_graph = build_research_graph()

app = FastAPI(title="Multi-Agent Research Assistant")
app.state.research_graph = research_graph
app.include_router(research_router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
