"""
LangChain ReAct sub-agents for the Bayer AI incident-response system.

Three agents are provided:
  - run_logger_agent   : investigates CloudWatch log events
  - run_metrics_agent  : analyses CloudWatch metric datapoints
  - run_deploy_agent   : checks recent CodeDeploy deployments

Each agent:
  - Receives ticket_id and incident_time as context in its prompt.
  - Uses the real AWS tools defined in tools.py.
  - Is instructed to emit structured JSON with keys:
      agent_name, findings, root_cause, confidence, queries_issued
  - The queries_issued list is collected by the orchestrator and embedded
    verbatim in the RCA Appendix section.
"""

import json
import logging
import os
import re
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from tools import (
    DEPLOY_AGENT_TOOLS,
    LOGGER_AGENT_TOOLS,
    METRICS_AGENT_TOOLS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM — OpenAI
# ---------------------------------------------------------------------------

_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=_OPENAI_MODEL,
        api_key=_OPENAI_API_KEY,
        temperature=0,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# ReAct prompt template
# ReAct agents need {tools}, {tool_names}, {agent_scratchpad}, and {input}.
# ---------------------------------------------------------------------------

_REACT_TEMPLATE = """{system_prompt}

You have access to the following tools:
{tools}

Tool names: {tool_names}

IMPORTANT OUTPUT RULES:
- After completing your investigation, you MUST output a JSON object as your Final Answer.
- The JSON must contain these keys:
    agent_name      : string — your agent role (e.g. "logger")
    findings        : list of strings — key observations
    root_cause      : string — most probable cause, or "" if unknown
    confidence      : float 0-1 — confidence in root_cause
    queries_issued  : list of strings — every tool call you made, formatted as
                       "<tool_name>(<args>)"
- Do NOT include markdown fences around the JSON.

Use this format:

Question: the input question you must answer
Thought: your reasoning
Action: the tool to use, one of [{tool_names}]
Action Input: the tool input
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to answer
Final Answer: <JSON object>

Begin!

Question: {input}
{agent_scratchpad}"""


def _make_prompt(system_prompt: str) -> PromptTemplate:
    return PromptTemplate.from_template(
        _REACT_TEMPLATE,
        partial_variables={"system_prompt": system_prompt},
    )


# ---------------------------------------------------------------------------
# JSON output parser
# ---------------------------------------------------------------------------

def _parse_agent_output(raw_output: Any, agent_name: str) -> dict:
    """
    Extract the JSON dict from whatever the agent returned.
    Falls back to a safe default if parsing fails.
    """
    text = raw_output if isinstance(raw_output, str) else str(raw_output)

    # Strip markdown fences if the model wrapped the JSON anyway
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Find the first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            result = json.loads(match.group(0))
            result.setdefault("agent_name", agent_name)
            result.setdefault("findings", [])
            result.setdefault("root_cause", "")
            result.setdefault("confidence", 0.0)
            result.setdefault("queries_issued", [])
            return result
        except json.JSONDecodeError:
            pass

    logger.warning("%s agent returned non-JSON output; using fallback.", agent_name)
    return {
        "agent_name": agent_name,
        "findings": [f"Raw output: {text[:500]}"],
        "root_cause": "",
        "confidence": 0.0,
        "queries_issued": [],
    }


# ---------------------------------------------------------------------------
# Generic agent runner
# ---------------------------------------------------------------------------

def _run_agent(
    agent_name: str,
    system_prompt: str,
    tools_list: list,
    user_message: str,
    max_iterations: int = 10,
) -> dict:
    llm = _build_llm()
    prompt = _make_prompt(system_prompt)

    agent = create_react_agent(llm=llm, tools=tools_list, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=True,
    )

    try:
        result = executor.invoke({"input": user_message})
        parsed = _parse_agent_output(result.get("output", ""), agent_name)

        # Reconstruct queries_issued from intermediate_steps if the model forgot
        if not parsed.get("queries_issued") and result.get("intermediate_steps"):
            auto_queries = []
            for action, _ in result["intermediate_steps"]:
                auto_queries.append(f"{action.tool}({action.tool_input})")
            parsed["queries_issued"] = auto_queries

        return parsed
    except Exception as exc:
        logger.error("%s agent execution failed: %s", agent_name, exc, exc_info=True)
        return {
            "agent_name": agent_name,
            "findings": [f"Agent execution error: {exc}"],
            "root_cause": "",
            "confidence": 0.0,
            "queries_issued": [],
        }


# ---------------------------------------------------------------------------
# Logger Agent
# ---------------------------------------------------------------------------

_LOGGER_SYSTEM = """You are the Logger Investigation Agent.
Your job is to fetch and analyse CloudWatch logs in the 15 minutes prior to an incident
and identify error patterns, exceptions, anomalies, and their likely cause.

When you call fetch_cloudwatch_logs:
  - Use the exact log_group provided in the question.
  - Use the exact incident_time provided.
  - Keep minutes_before=15 unless instructed otherwise.
  - Try a filter_pattern such as "ERROR" or "Exception" first,
    then fetch all logs if no results come back.

Think step-by-step. Record every tool call in queries_issued."""


def run_logger_agent(
    ticket_id: str,
    incident_time: str,
    log_group: str,
    error_code: str,
    incident_summary: str,
    kb_context: Optional[str] = None,
) -> dict:
    """
    Invoke the Logger sub-agent.

    Args:
        ticket_id: Unique incident identifier — passed as context.
        incident_time: ISO-8601 time of the alarm trigger.
        log_group: CloudWatch log group to investigate.
        error_code: Alarm name / error code from CloudWatch.
        incident_summary: Natural-language summary of the incident.
        kb_context: Optional relevant KB snippets.

    Returns:
        Dict with agent_name, findings, root_cause, confidence, queries_issued.
    """
    kb_section = f"\nRelevant knowledge-base context:\n{kb_context}" if kb_context else ""

    message = (
        f"[Ticket: {ticket_id}]\n"
        f"Incident Time: {incident_time}\n"
        f"Error Code / Alarm: {error_code}\n"
        f"Summary: {incident_summary}\n"
        f"Log Group to investigate: {log_group}"
        f"{kb_section}\n\n"
        "Fetch logs from 15 minutes before the incident and identify root causes."
    )
    return _run_agent(
        agent_name="logger",
        system_prompt=_LOGGER_SYSTEM,
        tools_list=LOGGER_AGENT_TOOLS,
        user_message=message,
    )


# ---------------------------------------------------------------------------
# Metrics Agent
# ---------------------------------------------------------------------------

_METRICS_SYSTEM = """You are the Metrics Investigation Agent.
Your job is to fetch CloudWatch metric statistics in the 15 minutes prior to an incident
and identify spikes, drops, or anomalies that indicate a performance or capacity problem.

When you call fetch_cloudwatch_metrics:
  - Use namespace, metric_name, and dimensions as provided or inferred.
  - Use the exact incident_time provided.
  - Relevant metrics to check include: Errors, Throttles, Duration, ConcurrentExecutions
    (for Lambda), CPUUtilization, RequestCount, TargetResponseTime (for ALB/ECS), etc.
  - Call the tool multiple times for different metrics if needed.

Think step-by-step. Record every tool call in queries_issued."""


def run_metrics_agent(
    ticket_id: str,
    incident_time: str,
    namespace: str,
    metric_names: list,
    dimensions: list,
    error_code: str,
    incident_summary: str,
    kb_context: Optional[str] = None,
) -> dict:
    """
    Invoke the Metrics sub-agent.

    Args:
        ticket_id: Unique incident identifier.
        incident_time: ISO-8601 time of the alarm trigger.
        namespace: CloudWatch namespace (e.g. 'AWS/Lambda').
        metric_names: List of metric names to inspect.
        dimensions: List of {Name, Value} dimension dicts (serialisable).
        error_code: Alarm name / error code.
        incident_summary: Natural-language summary.
        kb_context: Optional relevant KB snippets.

    Returns:
        Dict with agent_name, findings, root_cause, confidence, queries_issued.
    """
    kb_section = f"\nRelevant knowledge-base context:\n{kb_context}" if kb_context else ""
    dims_str = json.dumps(dimensions)

    message = (
        f"[Ticket: {ticket_id}]\n"
        f"Incident Time: {incident_time}\n"
        f"Error Code / Alarm: {error_code}\n"
        f"Summary: {incident_summary}\n"
        f"Namespace: {namespace}\n"
        f"Metrics to investigate: {', '.join(metric_names)}\n"
        f"Dimensions (JSON): {dims_str}"
        f"{kb_section}\n\n"
        "Fetch metric statistics from 15 minutes before the incident and diagnose the issue."
    )
    return _run_agent(
        agent_name="metrics",
        system_prompt=_METRICS_SYSTEM,
        tools_list=METRICS_AGENT_TOOLS,
        user_message=message,
    )


# ---------------------------------------------------------------------------
# Deploy Agent
# ---------------------------------------------------------------------------

_DEPLOY_SYSTEM = """You are the Deployment Intelligence Agent.
Your job is to check whether a recent CodeDeploy deployment may have caused or contributed
to the incident by fetching deployments from the 24 hours prior to the event.

When you call fetch_recent_deployments:
  - Use the application_name provided.
  - Use the exact incident_time.
  - Look for failed, partially-failed, or very-recent successful deployments
    that could correlate with the incident time.

Think step-by-step. Record every tool call in queries_issued."""


def run_deploy_agent(
    ticket_id: str,
    incident_time: str,
    application_name: str,
    error_code: str,
    incident_summary: str,
    kb_context: Optional[str] = None,
) -> dict:
    """
    Invoke the Deploy Intelligence sub-agent.

    Args:
        ticket_id: Unique incident identifier.
        incident_time: ISO-8601 time of the alarm trigger.
        application_name: CodeDeploy application name.
        error_code: Alarm name / error code.
        incident_summary: Natural-language summary.
        kb_context: Optional relevant KB snippets.

    Returns:
        Dict with agent_name, findings, root_cause, confidence, queries_issued.
    """
    kb_section = f"\nRelevant knowledge-base context:\n{kb_context}" if kb_context else ""

    message = (
        f"[Ticket: {ticket_id}]\n"
        f"Incident Time: {incident_time}\n"
        f"Error Code / Alarm: {error_code}\n"
        f"Summary: {incident_summary}\n"
        f"CodeDeploy Application: {application_name}"
        f"{kb_section}\n\n"
        "Fetch deployments from the last 24 hours and determine if a deployment caused this incident."
    )
    return _run_agent(
        agent_name="deploy",
        system_prompt=_DEPLOY_SYSTEM,
        tools_list=DEPLOY_AGENT_TOOLS,
        user_message=message,
    )
