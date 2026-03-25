"""
LangGraph StateGraph orchestrator for the Bayer AI incident-response system.

Graph topology:
  parse_alarm
    └─► kb_lookup
          └─► route  ──(kb_found)──► generate_rca
                    └──(no match)──► determine_agents
                                        └─► run_parallel_agents
                                               └─► correlate
                                                      └─► generate_rca
  generate_rca ──► store_kb ──► send_email ──► END

Entry point: run_graph(initial_state) -> IncidentState
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

import boto3
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import rca_generator
import sub_agents
from tools import send_sns_notification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REGION = os.environ.get("AWS_REGION", "us-east-1")

# OpenAI
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Bedrock KB + S3 (still AWS — only the LLM moves to OpenAI)
_KB_ID = os.environ.get("BEDROCK_KB_ID", "")
_KB_CONFIDENCE_THRESHOLD = float(os.environ.get("KB_CONFIDENCE_THRESHOLD", "0.6"))
_SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")

_kb_client = boto3.client("bedrock-agent-runtime", region_name=_REGION)
_s3_client = boto3.client("s3", region_name=_REGION)
_RCA_BUCKET = os.environ.get("RCA_S3_BUCKET", "")


# ---------------------------------------------------------------------------
# Shared LLM
# ---------------------------------------------------------------------------

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=_OPENAI_MODEL,
        api_key=_OPENAI_API_KEY,
        temperature=0,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Typed State
# ---------------------------------------------------------------------------

class IncidentState(TypedDict):
    # --- Alarm input ---
    ticket_id: str
    alarm_data: dict
    error_code: str
    incident_time: str          # ISO-8601 UTC
    incident_summary: str

    # --- KB lookup ---
    kb_snippets: List[str]
    kb_found: bool
    kb_result: Optional[str]    # Matched FAQ answer, if any

    # --- Dynamic agent selection ---
    active_agents: List[str]    # ["logger", "metrics", "deploy"] or subset
    agent_config: dict          # Per-agent parameters (log groups, namespaces, etc.)

    # --- Sub-agent outputs ---
    logger_output: Optional[dict]
    metrics_output: Optional[dict]
    deploy_output: Optional[dict]
    sub_agent_queries: List[str]  # Aggregated queries_issued from all agents

    # --- Correlation & report ---
    correlation: Optional[dict]
    rca_report: Optional[str]

    # --- Delivery ---
    email_sent: bool


# ---------------------------------------------------------------------------
# Helper: Bedrock KB retrieval (fixed — no invalid "type" field)
# ---------------------------------------------------------------------------

def retrieve_kb(query_text: str, max_results: int = 5) -> List[str]:
    """
    Query the Bedrock Knowledge Base and return a list of text snippets.
    The `retrievalQuery` dict contains only the `text` key — the previously
    present `"type": "TEXT"` field is not part of the valid Bedrock API.
    """
    if not _KB_ID:
        logger.warning("BEDROCK_KB_ID not set — skipping KB retrieval.")
        return []
    try:
        response = _kb_client.retrieve(
            knowledgeBaseId=_KB_ID,
            retrievalQuery={"text": query_text},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": max_results}
            },
        )
        snippets = [
            r["content"]["text"]
            for r in response.get("retrievalResults", [])
            if r.get("content", {}).get("text")
        ]
        return snippets
    except Exception as exc:
        logger.error("KB retrieval failed: %s", exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Node 1: parse_alarm
# ---------------------------------------------------------------------------

def parse_alarm(state: IncidentState) -> IncidentState:
    """
    Build a natural-language incident_summary from the raw alarm payload.
    The alarm_data is already parsed from the SNS message in lambda_handler.
    """
    alarm = state.get("alarm_data", {})
    error_code = state.get("error_code", "UNKNOWN")
    incident_time = state.get("incident_time", datetime.now(tz=timezone.utc).isoformat())

    alarm_desc = alarm.get("AlarmDescription", "")
    state_reason = alarm.get("NewStateReason", "")
    namespace = alarm.get("Trigger", {}).get("Namespace", "Unknown")
    metric = alarm.get("Trigger", {}).get("MetricName", "Unknown")

    summary = (
        f"CloudWatch alarm '{error_code}' triggered at {incident_time}. "
        f"Namespace: {namespace}. Metric: {metric}. "
        f"Reason: {state_reason}. "
        f"Description: {alarm_desc}."
    ).strip()

    state["incident_summary"] = summary
    state["kb_snippets"] = []
    state["kb_found"] = False
    state["kb_result"] = None
    state["active_agents"] = []
    state["agent_config"] = {}
    state["logger_output"] = None
    state["metrics_output"] = None
    state["deploy_output"] = None
    state["sub_agent_queries"] = []
    state["correlation"] = None
    state["rca_report"] = None
    state["email_sent"] = False

    logger.info("[%s] parse_alarm complete: %s", state["ticket_id"], summary[:120])
    return state


# ---------------------------------------------------------------------------
# Node 2: kb_lookup
# ---------------------------------------------------------------------------

def kb_lookup(state: IncidentState) -> IncidentState:
    """
    Query the Bedrock KB with the incident summary + error code.
    Use an LLM to grade whether the returned snippets actually answer the incident.
    Set kb_found=True and kb_result if a good answer is found.
    """
    query = f"{state['error_code']} {state['incident_summary']}"
    snippets = retrieve_kb(query, max_results=5)
    state["kb_snippets"] = snippets

    if not snippets:
        state["kb_found"] = False
        logger.info("[%s] KB returned no results.", state["ticket_id"])
        return state

    # Ask the LLM whether any snippet resolves this incident
    grading_prompt = (
        f"You are an SRE reviewing a knowledge base for a known solution.\n\n"
        f"Incident: {state['incident_summary']}\n\n"
        f"KB Snippets:\n"
        + "\n---\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
        + "\n\nDoes any snippet provide a clear, actionable solution to this incident? "
        "Reply with a JSON object: "
        '{"solution_found": true/false, "confidence": 0-1, "answer": "<snippet text or empty>"}'
        "\nOutput ONLY JSON."
    )

    try:
        llm = _llm()
        response = llm.invoke([HumanMessage(content=grading_prompt)])
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            grading = json.loads(match.group(0))
            if (
                grading.get("solution_found")
                and float(grading.get("confidence", 0)) >= _KB_CONFIDENCE_THRESHOLD
            ):
                state["kb_found"] = True
                state["kb_result"] = grading.get("answer", snippets[0])
                logger.info(
                    "[%s] KB solution found (confidence=%.2f).",
                    state["ticket_id"],
                    grading.get("confidence", 0),
                )
                return state
    except Exception as exc:
        logger.error("[%s] KB grading LLM call failed: %s", state["ticket_id"], exc)

    state["kb_found"] = False
    logger.info("[%s] No KB solution — proceeding to sub-agents.", state["ticket_id"])
    return state


# ---------------------------------------------------------------------------
# Routing function (conditional edge)
# ---------------------------------------------------------------------------

def route(state: IncidentState) -> str:
    return "generate_rca" if state.get("kb_found") else "determine_agents"


# ---------------------------------------------------------------------------
# Node 3: determine_agents
# ---------------------------------------------------------------------------

def determine_agents(state: IncidentState) -> IncidentState:
    """
    Ask the supervisor LLM which sub-agents to invoke and with what parameters,
    based on the alarm type and incident summary.

    The LLM returns a JSON with:
      active_agents: ["logger", "metrics", "deploy"]  (any subset)
      agent_config:
        logger:  { log_group: str }
        metrics: { namespace: str, metric_names: [str], dimensions: [{Name, Value}] }
        deploy:  { application_name: str }
    """
    prompt = (
        "You are an Autonomous Incident Commander deciding which investigation agents to deploy.\n\n"
        f"Ticket ID: {state['ticket_id']}\n"
        f"Error Code / Alarm: {state['error_code']}\n"
        f"Incident Summary: {state['incident_summary']}\n"
        f"Incident Time: {state['incident_time']}\n\n"
        "Available agents:\n"
        "  - logger  : fetches CloudWatch log events\n"
        "  - metrics : fetches CloudWatch metric statistics\n"
        "  - deploy  : checks recent CodeDeploy deployments\n\n"
        "Choose only the agents relevant to this incident. "
        "For each agent, provide the parameters it needs.\n\n"
        "Output ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "active_agents": ["logger", "metrics", "deploy"],\n'
        '  "reasoning": "why these agents",\n'
        '  "agent_config": {\n'
        '    "logger":  { "log_group": "/aws/lambda/my-function" },\n'
        '    "metrics": { "namespace": "AWS/Lambda", "metric_names": ["Errors","Duration"], '
        '"dimensions": [{"Name": "FunctionName", "Value": "my-function"}] },\n'
        '    "deploy":  { "application_name": "MyApp" }\n'
        "  }\n"
        "}\n"
        "Only include keys for agents you selected in active_agents."
    )

    try:
        llm = _llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            decision = json.loads(match.group(0))
            state["active_agents"] = decision.get("active_agents", ["logger", "metrics", "deploy"])
            state["agent_config"] = decision.get("agent_config", {})
            logger.info(
                "[%s] Supervisor selected agents: %s  reason: %s",
                state["ticket_id"],
                state["active_agents"],
                decision.get("reasoning", ""),
            )
            return state
    except Exception as exc:
        logger.error(
            "[%s] determine_agents LLM call failed: %s — defaulting to all agents.",
            state["ticket_id"],
            exc,
        )

    # Safe fallback
    state["active_agents"] = ["logger", "metrics", "deploy"]
    state["agent_config"] = {
        "logger": {"log_group": f"/aws/lambda/{state['error_code']}"},
        "metrics": {
            "namespace": "AWS/Lambda",
            "metric_names": ["Errors", "Duration", "Throttles"],
            "dimensions": [],
        },
        "deploy": {"application_name": state["error_code"]},
    }
    return state


# ---------------------------------------------------------------------------
# Node 4: run_parallel_agents
# ---------------------------------------------------------------------------

async def _run_agents_async(state: IncidentState) -> IncidentState:
    """Run selected sub-agents concurrently via asyncio."""
    active = state["active_agents"]
    cfg = state.get("agent_config", {})
    ticket_id = state["ticket_id"]
    incident_time = state["incident_time"]
    error_code = state["error_code"]
    incident_summary = state["incident_summary"]
    kb_context = "\n---\n".join(state.get("kb_snippets", [])) or None

    tasks = {}

    if "logger" in active:
        log_cfg = cfg.get("logger", {})
        tasks["logger"] = asyncio.to_thread(
            sub_agents.run_logger_agent,
            ticket_id=ticket_id,
            incident_time=incident_time,
            log_group=log_cfg.get("log_group", "/aws/lambda/default"),
            error_code=error_code,
            incident_summary=incident_summary,
            kb_context=kb_context,
        )

    if "metrics" in active:
        met_cfg = cfg.get("metrics", {})
        tasks["metrics"] = asyncio.to_thread(
            sub_agents.run_metrics_agent,
            ticket_id=ticket_id,
            incident_time=incident_time,
            namespace=met_cfg.get("namespace", "AWS/Lambda"),
            metric_names=met_cfg.get("metric_names", ["Errors"]),
            dimensions=met_cfg.get("dimensions", []),
            error_code=error_code,
            incident_summary=incident_summary,
            kb_context=kb_context,
        )

    if "deploy" in active:
        dep_cfg = cfg.get("deploy", {})
        tasks["deploy"] = asyncio.to_thread(
            sub_agents.run_deploy_agent,
            ticket_id=ticket_id,
            incident_time=incident_time,
            application_name=dep_cfg.get("application_name", error_code),
            error_code=error_code,
            incident_summary=incident_summary,
            kb_context=kb_context,
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    name_list = list(tasks.keys())

    all_queries: List[str] = []
    for name, result in zip(name_list, results):
        if isinstance(result, Exception):
            logger.error("[%s] %s agent raised: %s", ticket_id, name, result)
            output: dict = {
                "agent_name": name,
                "findings": [f"Agent error: {result}"],
                "root_cause": "",
                "confidence": 0.0,
                "queries_issued": [],
            }
        else:
            output = result  # type: ignore[assignment]

        all_queries.extend(output.get("queries_issued", []))

        if name == "logger":
            state["logger_output"] = output
        elif name == "metrics":
            state["metrics_output"] = output
        elif name == "deploy":
            state["deploy_output"] = output

    state["sub_agent_queries"] = all_queries
    logger.info(
        "[%s] All agents finished. Total queries issued: %d",
        ticket_id,
        len(all_queries),
    )
    return state


def run_parallel_agents(state: IncidentState) -> IncidentState:
    """Synchronous wrapper so LangGraph can call the async runner."""
    return asyncio.run(_run_agents_async(state))


# ---------------------------------------------------------------------------
# Node 5: correlate
# ---------------------------------------------------------------------------

def correlate(state: IncidentState) -> IncidentState:
    """
    Supervisor LLM call that correlates findings from all sub-agents and
    identifies the most probable root cause plus recommended fixes.
    """
    parts: List[str] = []

    if state.get("logger_output"):
        parts.append(f"Logger Agent:\n{json.dumps(state['logger_output'], indent=2)}")
    if state.get("metrics_output"):
        parts.append(f"Metrics Agent:\n{json.dumps(state['metrics_output'], indent=2)}")
    if state.get("deploy_output"):
        parts.append(f"Deploy Agent:\n{json.dumps(state['deploy_output'], indent=2)}")

    if not parts:
        state["correlation"] = {
            "root_cause": "No sub-agent data available.",
            "correlations": [],
            "recommended_fixes": [],
            "confidence": 0.0,
        }
        return state

    kb_section = ""
    if state.get("kb_snippets"):
        kb_section = "\n\nKnowledge Base context:\n" + "\n---\n".join(state["kb_snippets"])

    prompt = (
        f"You are an Autonomous Incident Commander correlating investigation results.\n\n"
        f"Ticket: {state['ticket_id']}\n"
        f"Incident: {state['incident_summary']}\n\n"
        + "\n\n".join(parts)
        + kb_section
        + "\n\nTasks:\n"
        "1. Identify causal relationships between agent findings.\n"
        "2. Determine the single most probable root cause.\n"
        "3. List recommended corrective actions (least-risky first).\n"
        "4. Note any conflicts or gaps in the data.\n\n"
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "root_cause": "...",\n'
        '  "correlations": ["..."],\n'
        '  "contributing_factors": ["..."],\n'
        '  "recommended_fixes": ["..."],\n'
        '  "confidence": 0.85,\n'
        '  "causal_chain": "Step1 → Step2 → Step3"\n'
        "}"
    )

    try:
        llm = _llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            state["correlation"] = json.loads(match.group(0))
            logger.info(
                "[%s] Correlation complete. Root cause: %s",
                state["ticket_id"],
                state["correlation"].get("root_cause", "")[:100],
            )
            return state
    except Exception as exc:
        logger.error("[%s] correlate LLM call failed: %s", state["ticket_id"], exc)

    state["correlation"] = {
        "root_cause": "Correlation failed — see sub-agent outputs.",
        "correlations": [],
        "recommended_fixes": [],
        "confidence": 0.0,
    }
    return state


# ---------------------------------------------------------------------------
# Node 6: generate_rca
# ---------------------------------------------------------------------------

def generate_rca(state: IncidentState) -> IncidentState:
    """
    Fill the RCA Markdown template using rca_generator and store in state.
    Works for both the KB fast-path (kb_found=True) and the full sub-agent path.
    """
    report = rca_generator.fill_template(state)
    state["rca_report"] = report
    logger.info("[%s] RCA report generated (%d chars).", state["ticket_id"], len(report))
    return state


# ---------------------------------------------------------------------------
# Node 7: store_kb
# ---------------------------------------------------------------------------

def store_kb(state: IncidentState) -> IncidentState:
    """
    Store the generated RCA report back to S3 (so it can be re-ingested into
    the Bedrock KB via a scheduled StartIngestionJob) and emit a LangSmith trace.

    S3 key: rca-reports/<ticket_id>.md
    """
    rca_md = state.get("rca_report", "")
    ticket_id = state["ticket_id"]

    # --- S3 storage ---
    if _RCA_BUCKET and rca_md:
        try:
            key = f"rca-reports/{ticket_id}.md"
            _s3_client.put_object(
                Bucket=_RCA_BUCKET,
                Key=key,
                Body=rca_md.encode("utf-8"),
                ContentType="text/markdown",
                Metadata={
                    "ticket_id": ticket_id,
                    "incident_time": state.get("incident_time", ""),
                    "error_code": state.get("error_code", ""),
                },
            )
            logger.info("[%s] RCA stored to s3://%s/%s", ticket_id, _RCA_BUCKET, key)
        except Exception as exc:
            logger.error("[%s] S3 store failed: %s", ticket_id, exc)
    else:
        logger.warning(
            "[%s] RCA_S3_BUCKET not configured — skipping S3 storage.", ticket_id
        )

    # --- LangSmith trace (best-effort) ---
    _emit_langsmith_trace(state)

    return state


def _emit_langsmith_trace(state: IncidentState) -> None:
    """Emit a run record to LangSmith if LANGSMITH_API_KEY is configured."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return
    try:
        from langsmith import Client as LangSmithClient

        ls = LangSmithClient(api_key=api_key)
        ls.create_run(
            name=f"IncidentRCA-{state['ticket_id']}",
            run_type="chain",
            inputs={
                "ticket_id": state["ticket_id"],
                "error_code": state["error_code"],
                "incident_summary": state["incident_summary"],
            },
            outputs={
                "root_cause": (state.get("correlation") or {}).get("root_cause", ""),
                "rca_report_length": len(state.get("rca_report") or ""),
                "kb_found": state.get("kb_found", False),
                "agents_used": state.get("active_agents", []),
            },
            project_name=os.environ.get("LANGSMITH_PROJECT", "bayer-incident-rca"),
        )
        logger.info("[%s] LangSmith trace emitted.", state["ticket_id"])
    except Exception as exc:
        logger.warning("[%s] LangSmith trace failed (non-fatal): %s", state["ticket_id"], exc)


# ---------------------------------------------------------------------------
# Node 8: send_email
# ---------------------------------------------------------------------------

def send_email(state: IncidentState) -> IncidentState:
    """
    Publish the RCA summary to the configured SNS topic.
    SNS delivers it to all subscribed endpoints (email, SMS, etc.) without
    requiring individual address verification.
    """
    ticket_id = state["ticket_id"]
    correlation = state.get("correlation") or {}
    root_cause = correlation.get("root_cause", "See full RCA report.")
    recommended_fixes = correlation.get("recommended_fixes", [])
    active_agents = state.get("active_agents", [])
    kb_found = state.get("kb_found", False)

    subject = f"[{ticket_id}] Incident RCA: {state['error_code']}"

    message = _build_sns_message(
        ticket_id=ticket_id,
        error_code=state["error_code"],
        incident_time=state["incident_time"],
        incident_summary=state["incident_summary"],
        root_cause=root_cause,
        recommended_fixes=recommended_fixes,
        agents_used=active_agents,
        kb_found=kb_found,
        rca_markdown=state.get("rca_report", "No RCA report generated."),
    )

    success = send_sns_notification(
        subject=subject,
        message=message,
        topic_arn=_SNS_TOPIC_ARN or None,
    )
    state["email_sent"] = success
    return state


def _build_sns_message(
    ticket_id: str,
    error_code: str,
    incident_time: str,
    incident_summary: str,
    root_cause: str,
    recommended_fixes: List[str],
    agents_used: List[str],
    kb_found: bool,
    rca_markdown: str,
) -> str:
    """Build a plain-text SNS message body containing the full RCA."""
    source_label = (
        "Knowledge Base (known issue)"
        if kb_found
        else f"Sub-agent investigation ({', '.join(agents_used) or 'none'})"
    )

    fixes_block = "\n".join(f"  - {f}" for f in recommended_fixes) or "  - See full RCA report."

    separator = "=" * 60

    return (
        f"{separator}\n"
        f"INCIDENT RCA ALERT\n"
        f"{separator}\n"
        f"Ticket ID        : {ticket_id}\n"
        f"Alarm / Error    : {error_code}\n"
        f"Incident Time    : {incident_time}\n"
        f"Investigation    : {source_label}\n"
        f"{separator}\n\n"
        f"SUMMARY\n{incident_summary}\n\n"
        f"ROOT CAUSE\n{root_cause}\n\n"
        f"RECOMMENDED FIXES\n{fixes_block}\n\n"
        f"{separator}\n"
        f"FULL RCA REPORT\n"
        f"{separator}\n\n"
        f"{rca_markdown}\n"
    )


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    graph = StateGraph(IncidentState)

    # Register nodes
    graph.add_node("parse_alarm", parse_alarm)
    graph.add_node("kb_lookup", kb_lookup)
    graph.add_node("determine_agents", determine_agents)
    graph.add_node("run_parallel_agents", run_parallel_agents)
    graph.add_node("correlate", correlate)
    graph.add_node("generate_rca", generate_rca)
    graph.add_node("store_kb", store_kb)
    graph.add_node("send_email", send_email)

    # Linear edges
    graph.add_edge("parse_alarm", "kb_lookup")
    graph.add_edge("determine_agents", "run_parallel_agents")
    graph.add_edge("run_parallel_agents", "correlate")
    graph.add_edge("correlate", "generate_rca")
    graph.add_edge("generate_rca", "store_kb")
    graph.add_edge("store_kb", "send_email")
    graph.add_edge("send_email", END)

    # Conditional edge: KB hit → skip agents; KB miss → run agents
    graph.add_conditional_edges(
        "kb_lookup",
        route,
        {
            "generate_rca": "generate_rca",
            "determine_agents": "determine_agents",
        },
    )

    # Entry point
    graph.set_entry_point("parse_alarm")

    return graph.compile()


# Compiled graph singleton — imported by lambda_handler
_GRAPH = None


def _get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_graph(initial_state: IncidentState) -> IncidentState:
    """
    Execute the full incident-response graph synchronously.

    Args:
        initial_state: Populated IncidentState dict (from lambda_handler).

    Returns:
        Final IncidentState after all nodes have executed.
    """
    graph = _get_graph()
    logger.info(
        "[%s] Starting incident graph for alarm '%s'.",
        initial_state["ticket_id"],
        initial_state["error_code"],
    )
    final_state: IncidentState = graph.invoke(initial_state)
    logger.info(
        "[%s] Graph complete. email_sent=%s, kb_found=%s.",
        final_state["ticket_id"],
        final_state.get("email_sent"),
        final_state.get("kb_found"),
    )
    return final_state
