"""
AWS Lambda entry point for the Bayer AI incident-response system.

Trigger: CloudWatch Alarm → SNS Topic → Lambda

The SNS message body is a JSON-serialised CloudWatch Alarm state-change
notification with this structure (subset):

{
  "AlarmName": "HighErrorRate-MyService",
  "AlarmDescription": "Error rate exceeded threshold",
  "NewStateValue": "ALARM",
  "NewStateReason": "Threshold Crossed: 1 out of the last 1 datapoints ...",
  "StateChangeTime": "2024-03-01T10:00:00.000+0000",
  "Region": "US East (N. Virginia)",
  "Trigger": {
    "MetricName": "Errors",
    "Namespace": "AWS/Lambda",
    "Dimensions": [{"name": "FunctionName", "value": "my-function"}]
  }
}

The handler:
1. Parses the SNS event.
2. Generates a unique ticket_id.
3. Builds the initial IncidentState.
4. Invokes orchestrator.run_graph(state).
5. Returns a summary dict to the caller (CloudWatch / SNS discard it, but
   useful for direct Lambda test invocations).

Environment variables (all required unless noted):
  AWS_REGION                — AWS region (default: us-east-1)
  BEDROCK_MODEL_ID          — Bedrock Claude model ID
  BEDROCK_KB_ID             — Bedrock Knowledge Base ID
  KB_CONFIDENCE_THRESHOLD   — Float 0-1, default 0.6
  RCA_S3_BUCKET             — S3 bucket for RCA storage
  ALERT_EMAIL_RECIPIENT     — SES destination email
  SES_SENDER                — SES verified sender email
  LANGSMITH_API_KEY         — (optional) LangSmith tracing
  LANGSMITH_PROJECT         — (optional) LangSmith project name
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import orchestrator
from orchestrator import IncidentState

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event parsers
# ---------------------------------------------------------------------------

def _parse_sns_event(event: dict) -> dict:
    """
    Extract the CloudWatch alarm payload from an SNS-wrapped Lambda event.
    Handles both SNS trigger events and direct test invocations.
    """
    # SNS → Lambda event structure
    if "Records" in event:
        records = event["Records"]
        if records and records[0].get("EventSource") == "aws:sns":
            sns_message = records[0]["Sns"]["Message"]
            return json.loads(sns_message)

    # Direct invocation (e.g. Lambda test console) — treat event as alarm payload
    if "AlarmName" in event:
        return event

    # Fallback: wrap unknown payload
    logger.warning("Unrecognised event shape — wrapping as generic alarm.")
    return {
        "AlarmName": event.get("detail-type", "UnknownAlarm"),
        "AlarmDescription": str(event),
        "NewStateValue": "ALARM",
        "NewStateReason": "Triggered by unknown event source.",
        "StateChangeTime": datetime.now(tz=timezone.utc).isoformat(),
        "Region": os.environ.get("AWS_REGION", "us-east-1"),
        "Trigger": {
            "MetricName": "Unknown",
            "Namespace": "Unknown",
            "Dimensions": [],
        },
    }


def _extract_error_code(alarm_data: dict) -> str:
    """
    Use AlarmName as the primary error/incident code.
    Falls back to MetricName if AlarmName is missing.
    """
    return (
        alarm_data.get("AlarmName")
        or alarm_data.get("Trigger", {}).get("MetricName")
        or "UNKNOWN"
    )


def _extract_incident_time(alarm_data: dict) -> str:
    """
    Return an ISO-8601 UTC timestamp for the alarm state change.
    Normalises CloudWatch's non-standard +0000 suffix.
    """
    raw = alarm_data.get("StateChangeTime", "")
    if raw:
        # CloudWatch uses "+0000" — Python fromisoformat needs "+00:00"
        raw = raw.replace("+0000", "+00:00").replace(".000+00:00", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
            return dt.isoformat()
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc).isoformat()


def _generate_ticket_id() -> str:
    """Generate a unique incident ticket ID: INC-<epoch_ms>."""
    return f"INC-{int(time.time() * 1000)}"


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict, context: Any) -> dict:
    """
    Main Lambda handler.

    Args:
        event: Raw Lambda event dict (SNS or direct invocation).
        context: Lambda context object (unused but required by AWS).

    Returns:
        Dict with ticket_id, status, root_cause, and email_sent flag.
    """
    logger.info("Lambda triggered. Event keys: %s", list(event.keys()))

    # --- Parse alarm ---
    try:
        alarm_data = _parse_sns_event(event)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to parse incoming event: %s", exc)
        return {"statusCode": 400, "body": f"Event parsing failed: {exc}"}

    ticket_id = _generate_ticket_id()
    error_code = _extract_error_code(alarm_data)
    incident_time = _extract_incident_time(alarm_data)

    logger.info(
        "New incident: ticket=%s  alarm=%s  time=%s",
        ticket_id,
        error_code,
        incident_time,
    )

    # --- Build initial state ---
    initial_state: IncidentState = {
        # Populated by lambda_handler
        "ticket_id": ticket_id,
        "alarm_data": alarm_data,
        "error_code": error_code,
        "incident_time": incident_time,
        # Populated by graph nodes — initialise to empty
        "incident_summary": "",
        "kb_snippets": [],
        "kb_found": False,
        "kb_result": None,
        "active_agents": [],
        "agent_config": {},
        "logger_output": None,
        "metrics_output": None,
        "deploy_output": None,
        "sub_agent_queries": [],
        "correlation": None,
        "rca_report": None,
        "email_sent": False,
    }

    # --- Run the incident-response graph ---
    try:
        final_state = orchestrator.run_graph(initial_state)
    except Exception as exc:
        logger.error(
            "[%s] Graph execution failed: %s", ticket_id, exc, exc_info=True
        )
        return {
            "statusCode": 500,
            "ticket_id": ticket_id,
            "error": str(exc),
        }

    # --- Build response ---
    correlation = final_state.get("correlation") or {}
    response = {
        "statusCode": 200,
        "ticket_id": ticket_id,
        "error_code": error_code,
        "incident_time": incident_time,
        "kb_found": final_state.get("kb_found", False),
        "agents_used": final_state.get("active_agents", []),
        "root_cause": correlation.get("root_cause", final_state.get("kb_result", "N/A")),
        "rca_report_length": len(final_state.get("rca_report") or ""),
        "email_sent": final_state.get("email_sent", False),
    }

    logger.info(
        "[%s] Incident resolved. root_cause=%s  email_sent=%s",
        ticket_id,
        response["root_cause"][:80],
        response["email_sent"],
    )
    return response
