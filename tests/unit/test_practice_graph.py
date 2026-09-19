"""Unit tests for the practice LangGraph graph and its node functions."""

from app.graph.practice_graph import build_practice_graph
from app.graph.practice_nodes import say_hello_node, shout_node
from app.graph.practice_state import PracticeState


def test_graph_invoke_produces_shouted_greeting() -> None:
    graph = build_practice_graph()
    result = graph.invoke(PracticeState(name="Test"))
    assert result["message"] == "HELLO, TEST!"


def test_say_hello_node_sets_greeting() -> None:
    state = PracticeState(name="Test")
    result = say_hello_node(state)
    assert result.message == "Hello, Test!"


def test_shout_node_uppercases_message() -> None:
    state = PracticeState(name="Test", message="hello, test!")
    result = shout_node(state)
    assert result.message == "HELLO, TEST!"
