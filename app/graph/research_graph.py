"""Wires the Planner, Researcher, Supervisor, and Writer nodes into one graph.

Assembly only -- no business logic lives here. Each node's behavior is
defined and tested in its own agent module.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.supervisor import supervisor_node
from app.agents.writer import writer_node
from app.graph.routing import route_after_supervisor
from app.schemas.research import ResearchState


def build_research_graph() -> CompiledStateGraph:
    """Build and compile the end-to-end research graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("writer", END)

    return graph.compile()
