"""Streamlit frontend for the Multi-Agent Research Assistant.

A thin UI over the existing FastAPI app: POST /research to start a run, then
poll GET /research/{run_id} (and GET /research/{run_id}/trace, to derive a
more specific "which agent is running now" message) until it finishes. No
agent/business/graph logic lives here -- this only ever talks to the API
over HTTP, exactly like the curl examples in README.md.
"""

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# Generous on purpose: POST /research's own handler writes to a remotely
# hosted Postgres (Supabase) before it can respond, and a too-tight client
# timeout there would surface as a misleading "connection error" for what's
# actually just normal network/cold-start latency, not a real failure.
REQUEST_TIMEOUT_SECONDS = 20
POLL_INTERVAL_SECONDS = 2.5

# docker-compose's `depends_on: app` only orders container *start*, not
# *readiness* -- if this page loads and a question is submitted before
# uvicorn inside the app container is actually accepting connections yet,
# the very first request would otherwise fail with a raw connection error.
# A few short retries absorb that startup window without the user seeing it.
STARTUP_RETRY_ATTEMPTS = 5
STARTUP_RETRY_DELAY_SECONDS = 2

# GET /research/{id} only reports "pending"/"running"/"completed"/"failed" --
# it doesn't say which agent is currently active. GET /research/{id}/trace
# does report one row per node *once that node has finished*, so the most
# recently completed node name is used to derive a more specific in-progress
# message than "Processing..." wherever that's actually inferable.
_STAGE_MESSAGE_AFTER_NODE = {
    None: "Planning...",
    "planner": "Researching...",
    "researcher": "Evaluating findings...",
    "supervisor": "Continuing...",
    "writer": "Finalizing...",
}


def _post_research(question: str) -> dict:
    """POST /research, retrying through the container-startup window."""
    last_exc: Exception | None = None
    for attempt in range(STARTUP_RETRY_ATTEMPTS):
        try:
            response = requests.post(
                f"{API_BASE_URL}/research",
                json={"question": question},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < STARTUP_RETRY_ATTEMPTS - 1:
                time.sleep(STARTUP_RETRY_DELAY_SECONDS)
    assert last_exc is not None
    raise last_exc


def _get_status(run_id: str) -> dict:
    response = requests.get(f"{API_BASE_URL}/research/{run_id}", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _get_current_stage_message(run_id: str) -> str:
    """Best-effort "which agent is running now" message, derived from the
    trace endpoint's most recently completed node. Falls back to a generic
    message if the trace can't be fetched -- this is a progress nicety, not
    something a failure here should ever interrupt the main polling loop for.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/research/{run_id}/trace", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        steps = response.json().get("steps", [])
    except requests.exceptions.RequestException:
        return "Processing..."
    last_node = steps[-1]["node_name"] if steps else None
    return _STAGE_MESSAGE_AFTER_NODE.get(last_node, "Processing...")


def _extract_sources_section(report_markdown: str) -> str | None:
    """Pull out the report's "## Sources" section, if present, so it can be
    surfaced on its own in addition to the full rendered report."""
    marker = "## Sources"
    index = report_markdown.find(marker)
    if index == -1:
        return None
    return report_markdown[index:].strip()


st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎")
st.title("Multi-Agent Research Assistant")
st.caption(f"API: {API_BASE_URL}")

question = st.text_input(
    "Research question",
    placeholder="e.g. What are the benefits and risks of remote work?",
)

if st.button("Start Research", type="primary", disabled=not question.strip()):
    status_placeholder = st.empty()

    try:
        created = _post_research(question)
    except requests.exceptions.ConnectionError:
        status_placeholder.error(
            f"Could not reach the API at {API_BASE_URL}. Is the app service running?"
        )
    except requests.exceptions.RequestException as exc:
        status_placeholder.error(f"Failed to start the research run: {exc}")
    else:
        run_id = created["run_id"]
        status_placeholder.info(f"⏳ Run started (id: `{run_id}`). Planning...")

        final_record = None
        connection_lost = False
        while True:
            try:
                record = _get_status(run_id)
            except requests.exceptions.RequestException as exc:
                status_placeholder.error(f"Lost connection while checking progress: {exc}")
                connection_lost = True
                break

            if record["status"] not in ("pending", "running"):
                final_record = record
                break

            stage_message = _get_current_stage_message(run_id) if record["status"] == "running" else "Queued..."
            status_placeholder.info(f"⏳ {stage_message}")
            time.sleep(POLL_INTERVAL_SECONDS)

        if not connection_lost and final_record is not None:
            status_placeholder.empty()

            if final_record["status"] == "completed":
                st.success("Research complete.")
                report = final_record.get("final_report") or "*(no report content)*"
                st.markdown(report)

                sources_section = _extract_sources_section(report)
                if sources_section:
                    with st.expander("Sources", expanded=False):
                        st.markdown(sources_section)

            elif final_record["status"] == "failed":
                st.error(f"Research run failed: {final_record.get('error') or 'Unknown error'}")

            else:
                st.warning(f"Run ended in unexpected status: {final_record['status']!r}")
