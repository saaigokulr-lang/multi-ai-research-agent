"""Node functions for the practice LangGraph graph."""

from app.core.logging import get_logger
from app.graph.practice_state import PracticeState

logger = get_logger(__name__)


def say_hello_node(state: PracticeState) -> PracticeState:
    """Set a greeting message for state.name."""
    logger.info("entering say_hello_node")
    state.message = f"Hello, {state.name}!"
    logger.info("exiting say_hello_node")
    return state


def shout_node(state: PracticeState) -> PracticeState:
    """Upper-case the current message."""
    logger.info("entering shout_node")
    state.message = state.message.upper()
    logger.info("exiting shout_node")
    return state
