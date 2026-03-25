"""
orch_2.py — SUPERSEDED

This file previously contained a draft LangGraph-style orchestrator.
All logic has been migrated and corrected in orchestrator.py.

Issues that were fixed during migration:
  - `retrieve_kb` referenced undefined `client` (now uses `kb_client`)
  - `from langgraph.graph import Graph, Node` — these classes do not exist in
    modern LangGraph; replaced with `StateGraph` and `END`
  - Model ID `"sonnet-lite-v1"` is not a real Bedrock model ARN; replaced with
    `anthropic.claude-3-5-sonnet-20241022-v2:0` (configurable via env var)
  - Sub-agent wrappers were pure stubs; replaced with real LangChain ReAct agents
  - `kb_client.add_document(...)` does not exist in boto3; replaced with
    S3 `put_object` + Bedrock `StartIngestionJob`
  - No Lambda entry point; added `lambda_handler.py`
  - No `ticket_id` context; now threaded through every node and sub-agent call
  - No dynamic agent selection; now performed by `determine_agents` node
  - `retrievalQuery` contained invalid `"type": "TEXT"` field; removed

Public API (re-exported from orchestrator for backward compatibility):
  - run_graph(initial_state)  — full graph entry point
  - retrieve_kb(query, max)   — KB retrieval helper
  - IncidentState             — typed state dict
"""

from orchestrator import IncidentState, retrieve_kb, run_graph  # noqa: F401

__all__ = ["run_graph", "retrieve_kb", "IncidentState"]
