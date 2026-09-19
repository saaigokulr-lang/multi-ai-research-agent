"""Minimal LangGraph graph used to practice node/edge wiring.

Not the real project graph -- just START -> say_hello_node -> shout_node -> END.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.practice_nodes import say_hello_node, shout_node
from app.graph.practice_state import PracticeState


def build_practice_graph() -> CompiledStateGraph:
    """Build and compile the practice graph."""
    graph = StateGraph(PracticeState)
    graph.add_node("say_hello_node", say_hello_node)
    graph.add_node("shout_node", shout_node)
    graph.add_edge(START, "say_hello_node")
    graph.add_edge("say_hello_node", "shout_node")
    graph.add_edge("shout_node", END)
    return graph.compile()
