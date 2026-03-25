"""
AWS tool implementations for the Bayer AI incident-response system.

Each tool is decorated with @tool so LangChain ReAct agents can call them
by name. Tools cover CloudWatch Logs, CloudWatch Metrics, CodeDeploy, and SES.
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Boto3 client singletons — region resolved from env or falls back to us-east-1
# ---------------------------------------------------------------------------
_REGION = os.environ.get("AWS_REGION", "us-east-1")

_logs_client = None
_cw_client = None
_deploy_client = None
_sns_client = None


def _get_logs_client():
    global _logs_client
    if _logs_client is None:
        _logs_client = boto3.client("logs", region_name=_REGION)
    return _logs_client


def _get_cw_client():
    global _cw_client
    if _cw_client is None:
        _cw_client = boto3.client("cloudwatch", region_name=_REGION)
    return _cw_client


def _get_deploy_client():
    global _deploy_client
    if _deploy_client is None:
        _deploy_client = boto3.client("codedeploy", region_name=_REGION)
    return _deploy_client


def _get_sns_client():
    global _sns_client
    if _sns_client is None:
        _sns_client = boto3.client("sns", region_name=_REGION)
    return _sns_client


def _parse_incident_time(incident_time: str) -> datetime:
    """
    Parse an ISO-8601 timestamp string into an aware UTC datetime.
    Handles both Z-suffix and +00:00 offset forms.
    """
    ts = incident_time.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# CloudWatch Logs tool
# ---------------------------------------------------------------------------

@tool
def fetch_cloudwatch_logs(
    log_group: str,
    incident_time: str,
    minutes_before: int = 15,
    filter_pattern: str = "",
) -> str:
    """
    Fetch CloudWatch log events from the given log group in the window
    [incident_time - minutes_before, incident_time].

    Args:
        log_group: The CloudWatch log group name (e.g. '/aws/lambda/my-function').
        incident_time: ISO-8601 timestamp of the incident (e.g. '2024-03-01T10:00:00Z').
        minutes_before: How many minutes before the incident to look back (default 15).
        filter_pattern: Optional CloudWatch filter pattern string.

    Returns:
        JSON string with a list of log event dicts (timestamp, message).
    """
    try:
        end_dt = _parse_incident_time(incident_time)
        start_dt = end_dt - timedelta(minutes=minutes_before)

        end_ms = int(end_dt.timestamp() * 1000)
        start_ms = int(start_dt.timestamp() * 1000)

        kwargs: dict = {
            "logGroupName": log_group,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 500,
        }
        if filter_pattern:
            kwargs["filterPattern"] = filter_pattern

        paginator = _get_logs_client().get_paginator("filter_log_events")
        events = []
        for page in paginator.paginate(**kwargs):
            for ev in page.get("events", []):
                events.append(
                    {
                        "timestamp": datetime.fromtimestamp(
                            ev["timestamp"] / 1000, tz=timezone.utc
                        ).isoformat(),
                        "message": ev.get("message", ""),
                        "logStreamName": ev.get("logStreamName", ""),
                    }
                )
            if len(events) >= 500:
                break

        return json.dumps(
            {
                "log_group": log_group,
                "window_start": start_dt.isoformat(),
                "window_end": end_dt.isoformat(),
                "event_count": len(events),
                "events": events,
            }
        )
    except Exception as exc:
        logger.error("fetch_cloudwatch_logs failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "log_group": log_group})


# ---------------------------------------------------------------------------
# CloudWatch Metrics tool
# ---------------------------------------------------------------------------

@tool
def fetch_cloudwatch_metrics(
    namespace: str,
    metric_name: str,
    incident_time: str,
    dimensions: str = "[]",
    minutes_before: int = 15,
    period_seconds: int = 60,
    statistic: str = "Average",
) -> str:
    """
    Fetch CloudWatch metric statistics for a time window prior to the incident.

    Args:
        namespace: CloudWatch namespace (e.g. 'AWS/Lambda').
        metric_name: Metric name (e.g. 'Errors', 'Duration', 'Throttles').
        incident_time: ISO-8601 timestamp of the incident.
        dimensions: JSON-encoded list of {Name, Value} dimension dicts
                    (e.g. '[{"Name":"FunctionName","Value":"my-fn"}]').
        minutes_before: Window size in minutes (default 15).
        period_seconds: Aggregation period in seconds (default 60).
        statistic: One of Average | Sum | Maximum | Minimum | SampleCount.

    Returns:
        JSON string with metric datapoints and metadata.
    """
    try:
        end_dt = _parse_incident_time(incident_time)
        start_dt = end_dt - timedelta(minutes=minutes_before)

        parsed_dims = json.loads(dimensions) if dimensions else []

        response = _get_cw_client().get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=parsed_dims,
            StartTime=start_dt,
            EndTime=end_dt,
            Period=period_seconds,
            Statistics=[statistic],
        )

        datapoints = sorted(
            [
                {
                    "timestamp": dp["Timestamp"].isoformat(),
                    statistic.lower(): dp.get(statistic),
                    "unit": dp.get("Unit", ""),
                }
                for dp in response.get("Datapoints", [])
            ],
            key=lambda x: x["timestamp"],
        )

        return json.dumps(
            {
                "namespace": namespace,
                "metric_name": metric_name,
                "dimensions": parsed_dims,
                "window_start": start_dt.isoformat(),
                "window_end": end_dt.isoformat(),
                "period_seconds": period_seconds,
                "statistic": statistic,
                "datapoint_count": len(datapoints),
                "datapoints": datapoints,
            }
        )
    except Exception as exc:
        logger.error("fetch_cloudwatch_metrics failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "namespace": namespace, "metric_name": metric_name})


# ---------------------------------------------------------------------------
# CodeDeploy recent deployments tool
# ---------------------------------------------------------------------------

@tool
def fetch_recent_deployments(
    application_name: str,
    incident_time: str,
    hours_before: int = 24,
    deployment_group: Optional[str] = None,
) -> str:
    """
    Fetch CodeDeploy deployments for an application in the hours prior to the incident.

    Args:
        application_name: CodeDeploy application name.
        incident_time: ISO-8601 timestamp of the incident.
        hours_before: Look-back window in hours (default 24).
        deployment_group: Optional deployment group name to filter on.

    Returns:
        JSON string with a list of deployment summaries.
    """
    try:
        end_dt = _parse_incident_time(incident_time)
        start_dt = end_dt - timedelta(hours=hours_before)

        cd = _get_deploy_client()

        list_kwargs: dict = {
            "applicationName": application_name,
            "createTimeRange": {
                "start": start_dt,
                "end": end_dt,
            },
        }
        if deployment_group:
            list_kwargs["deploymentGroupName"] = deployment_group

        deployment_ids: list = []
        paginator = cd.get_paginator("list_deployments")
        for page in paginator.paginate(**list_kwargs):
            deployment_ids.extend(page.get("deployments", []))

        deployments = []
        for dep_id in deployment_ids[:20]:
            try:
                info = cd.get_deployment(deploymentId=dep_id)["deploymentInfo"]
                deployments.append(
                    {
                        "deploymentId": dep_id,
                        "status": info.get("status"),
                        "createTime": info.get("createTime", "").isoformat()
                        if hasattr(info.get("createTime", ""), "isoformat")
                        else str(info.get("createTime", "")),
                        "completeTime": info.get("completeTime", "").isoformat()
                        if hasattr(info.get("completeTime", ""), "isoformat")
                        else str(info.get("completeTime", "")),
                        "deploymentGroupName": info.get("deploymentGroupName"),
                        "revision": info.get("revision", {}),
                        "errorInformation": info.get("errorInformation"),
                        "description": info.get("description", ""),
                    }
                )
            except Exception as inner_exc:
                logger.warning("Could not fetch deployment %s: %s", dep_id, inner_exc)

        return json.dumps(
            {
                "application_name": application_name,
                "window_start": start_dt.isoformat(),
                "window_end": end_dt.isoformat(),
                "deployment_count": len(deployments),
                "deployments": deployments,
            }
        )
    except Exception as exc:
        logger.error("fetch_recent_deployments failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "application_name": application_name})


# ---------------------------------------------------------------------------
# SNS notification sender (not a LangChain tool — called directly by the graph)
# ---------------------------------------------------------------------------

def send_sns_notification(
    subject: str,
    message: str,
    topic_arn: Optional[str] = None,
) -> bool:
    """
    Publish an incident alert to an SNS topic.

    SNS delivers the message to all topic subscribers (email, SMS, Lambda, etc.)
    without requiring individual address verification — unlike SES.

    Args:
        subject: Message subject (shown in email subject line by SNS, max 100 chars).
        message: Plain-text message body.
        topic_arn: SNS topic ARN. Falls back to SNS_TOPIC_ARN env var.

    Returns:
        True on success, False on failure.
    """
    arn = topic_arn or os.environ.get("SNS_TOPIC_ARN", "")
    if not arn:
        logger.error("SNS_TOPIC_ARN is not set — cannot publish notification.")
        return False

    # SNS subject line is capped at 100 characters
    safe_subject = subject[:100]

    try:
        response = _get_sns_client().publish(
            TopicArn=arn,
            Subject=safe_subject,
            Message=message,
        )
        logger.info(
            "SNS notification published | MessageId: %s | subject: %s",
            response.get("MessageId"),
            safe_subject,
        )
        return True
    except Exception as exc:
        logger.error("send_sns_notification failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Exported tool lists per agent type
# ---------------------------------------------------------------------------

LOGGER_AGENT_TOOLS = [fetch_cloudwatch_logs]
METRICS_AGENT_TOOLS = [fetch_cloudwatch_metrics]
DEPLOY_AGENT_TOOLS = [fetch_recent_deployments]
