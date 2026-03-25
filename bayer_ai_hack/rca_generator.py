"""
RCA report generator for the Bayer AI incident-response system.

fill_template(state) takes a fully populated IncidentState and returns a
complete Markdown RCA report based on the template in report_gen.ipynb.

Two paths:
  - KB fast-path (state["kb_found"] = True):
      Uses the KB answer as the root cause and produces a streamlined report.
  - Full sub-agent path:
      Populates every section from sub-agent and correlation outputs,
      and appends a dedicated Appendix section listing every query issued.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from orchestrator import IncidentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe(value: Any, fallback: str = "N/A") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _bullet_list(items: List[str], indent: str = "- ") -> str:
    if not items:
        return f"{indent}N/A"
    return "\n".join(f"{indent}{item}" for item in items)


def _checklist(items: List[str]) -> str:
    if not items:
        return "- [ ] N/A"
    return "\n".join(f"- [ ] {item}" for item in items)


def _agent_section(output: Optional[dict], label: str) -> str:
    if not output:
        return (
            f"### {label}\n"
            "- Key Findings:\n  - Agent was not invoked for this incident.\n"
            "- Suspected Cause:\n  - N/A\n"
            "- Confidence:\n  - N/A\n"
        )
    findings = _bullet_list(output.get("findings", []), indent="  - ")
    root_cause = _safe(output.get("root_cause"), "Not determined")
    confidence = output.get("confidence", 0.0)
    return (
        f"### {label}\n"
        f"- Key Findings:\n{findings}\n"
        f"- Suspected Cause:\n  - {root_cause}\n"
        f"- Confidence:\n  - {confidence:.0%}\n"
    )


def _queries_appendix(queries: List[str]) -> str:
    if not queries:
        return "_No tool queries were issued._"
    lines = []
    for i, q in enumerate(queries, 1):
        lines.append(f"{i}. `{q}`")
    return "\n".join(lines)


def _timeline_from_state(state: Dict[str, Any]) -> str:
    rows = [
        "| Time | Event |",
        "|------|-------|",
        f"| {_safe(state.get('incident_time'))} | CloudWatch alarm triggered: {_safe(state.get('error_code'))} |",
    ]

    if state.get("kb_found"):
        rows.append(f"| {_now_utc()} | KB solution found — fast-path RCA generated |")
    else:
        active = state.get("active_agents", [])
        if active:
            rows.append(f"| {_now_utc()} | Sub-agents invoked: {', '.join(active)} |")
        if state.get("correlation"):
            rows.append(f"| {_now_utc()} | Supervisor correlation complete |")
        rows.append(f"| {_now_utc()} | RCA report generated |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main template filler
# ---------------------------------------------------------------------------

def fill_template(state: Dict[str, Any]) -> str:
    """
    Populate the RCA Markdown template with data from the incident state.

    Args:
        state: IncidentState dict (fully populated after graph execution).

    Returns:
        Complete Markdown RCA report as a string.
    """
    ticket_id = _safe(state.get("ticket_id"), "INC-UNKNOWN")
    error_code = _safe(state.get("error_code"), "UNKNOWN")
    incident_time = _safe(state.get("incident_time"), _now_utc())
    incident_summary = _safe(state.get("incident_summary"), "No summary available.")
    kb_found: bool = state.get("kb_found", False)
    alarm_data: dict = state.get("alarm_data", {})

    # Severity heuristic: use alarm data if available, else default SEV-2
    severity = alarm_data.get("NewStateValue", "ALARM")
    severity_label = "SEV-1" if severity == "ALARM" else "SEV-2"

    # Correlation block (sub-agent path)
    correlation: dict = state.get("correlation") or {}
    root_cause = _safe(correlation.get("root_cause") or state.get("kb_result"), "Not determined")
    correlations_list: List[str] = correlation.get("correlations", [])
    contributing_factors: List[str] = correlation.get("contributing_factors", [])
    recommended_fixes: List[str] = correlation.get("recommended_fixes", [])
    causal_chain = _safe(correlation.get("causal_chain"), "Not determined")
    confidence_score = correlation.get("confidence", 0.0)

    # Sub-agent outputs
    logger_output: Optional[dict] = state.get("logger_output")
    metrics_output: Optional[dict] = state.get("metrics_output")
    deploy_output: Optional[dict] = state.get("deploy_output")
    sub_agent_queries: List[str] = state.get("sub_agent_queries", [])
    active_agents: List[str] = state.get("active_agents", [])

    # Executive summary (short paragraph)
    exec_summary = _build_exec_summary(
        kb_found=kb_found,
        error_code=error_code,
        incident_time=incident_time,
        root_cause=root_cause,
        recommended_fixes=recommended_fixes,
    )

    # Alarm metadata for detection section
    alarm_desc = alarm_data.get("AlarmDescription", "N/A")
    state_reason = alarm_data.get("NewStateReason", "N/A")
    namespace = alarm_data.get("Trigger", {}).get("Namespace", "N/A")
    metric = alarm_data.get("Trigger", {}).get("MetricName", "N/A")

    # Source label
    investigation_source = (
        "Knowledge Base (known issue — fast-path)"
        if kb_found
        else f"Sub-agent investigation ({', '.join(active_agents) or 'none'})"
    )

    report = f"""# Root Cause Analysis (RCA) Report

## Incident Metadata
- **Incident ID**: {ticket_id}
- **Date**: {incident_time[:10] if len(incident_time) >= 10 else incident_time}
- **Time Window**: {incident_time} → {_now_utc()}
- **Severity**: {severity_label}
- **Status**: Resolved
- **Reported By**: CloudWatch Alarm — {error_code}
- **Investigation Source**: {investigation_source}

---

## Executive Summary
{exec_summary}

---

## Incident Description
- **What went wrong**: {error_code} alarm triggered in CloudWatch.
- **Symptoms observed**: {state_reason}
- **Alarm description**: {alarm_desc}
- **Namespace / Metric**: {namespace} / {metric}
- **Systems / services affected**: See sub-agent findings below.

---

## Impact Assessment
- **User Impact**: To be determined from log/metrics analysis.
- **System Impact**: {state_reason}
- **Business Impact**: Severity {severity_label} — SLA review required if SEV-1.

---

## Timeline of Events
{_timeline_from_state(state)}

---

## Detection & Alerting
- **How detected**: CloudWatch alarm `{error_code}` transitioned to `{severity}` state.
- **Alerts fired**: `{error_code}`
- **Detection delay**: Measured from alarm trigger at {incident_time}.

---

## Investigation Summary

{_agent_section(logger_output, "Logger Agent Insights")}
---

{_agent_section(metrics_output, "Metrics Agent Insights")}
---

{_agent_section(deploy_output, "Deploy Intelligence Insights")}

---

## Correlation Analysis
- **Cross-agent relationships**:
{_bullet_list(correlations_list) if correlations_list else "- KB fast-path — no cross-agent correlation performed." if kb_found else "- N/A"}
- **Contributing factors**:
{_bullet_list(contributing_factors)}
- **Most plausible causal chain**:
  - {causal_chain}

---

## Root Cause
**Primary Root Cause:**
- {root_cause}

**Contributing Factors:**
{_bullet_list(contributing_factors)}

**Confidence Score:** {f"{confidence_score:.0%}" if confidence_score else "N/A"}

---

## Resolution
- Resolution actions are listed under Corrective Actions.
- Resolved at: {_now_utc()}

---

## Corrective Actions

### Immediate Fixes
{_checklist(recommended_fixes[:3] if recommended_fixes else [])}

### Preventative Measures
{_checklist(recommended_fixes[3:6] if len(recommended_fixes) > 3 else [])}

### Long-Term Improvements
- [ ] Add monitoring/alerting for early detection of similar patterns.
- [ ] Review deployment runbooks to include rollback criteria.
- [ ] Add automated canary testing post-deployment.

---

## Lessons Learned
- **What went well**: CloudWatch alarm detected the issue promptly.
- **What didn't**: {root_cause[:120] if root_cause != "Not determined" else "Root cause not yet fully determined."}
- **What can be improved**: Review corrective actions above.

---

## Follow-Up Actions
| Owner | Deadline | Action |
|-------|----------|--------|
| On-call SRE | 24 hours | Implement immediate fixes |
| Platform team | 1 week | Implement preventative measures |
| Engineering | 2 weeks | Long-term improvements |

---

## Appendix

### Tool Queries Issued by Sub-Agents
{_queries_appendix(sub_agent_queries)}

### Raw Sub-Agent Outputs

<details>
<summary>Logger Agent Output</summary>

```json
{json.dumps(logger_output, indent=2) if logger_output else "Not invoked"}
```

</details>

<details>
<summary>Metrics Agent Output</summary>

```json
{json.dumps(metrics_output, indent=2) if metrics_output else "Not invoked"}
```

</details>

<details>
<summary>Deploy Agent Output</summary>

```json
{json.dumps(deploy_output, indent=2) if deploy_output else "Not invoked"}
```

</details>

### Knowledge Base Snippets Used
{_bullet_list(state.get("kb_snippets", []))}

---

## Final Summary (For Non-Technical Stakeholders)

- **What went wrong**: The `{error_code}` service experienced an incident at {incident_time[:19].replace("T", " ")} UTC.
- **Why it happened**: {root_cause}
- **How it was fixed**: {recommended_fixes[0] if recommended_fixes else "Investigation complete — see corrective actions."}
- **How we will prevent it**: Monitoring improvements and deployment safeguards are being implemented.

---
_Report generated automatically by the Bayer AI Incident Response System._
_Ticket: {ticket_id} | Generated: {_now_utc()}_
"""
    return report


# ---------------------------------------------------------------------------
# Executive summary builder
# ---------------------------------------------------------------------------

def _build_exec_summary(
    kb_found: bool,
    error_code: str,
    incident_time: str,
    root_cause: str,
    recommended_fixes: List[str],
) -> str:
    fix_preview = recommended_fixes[0] if recommended_fixes else "corrective actions identified"
    if kb_found:
        return (
            f"A known incident pattern matching `{error_code}` was identified in the knowledge base. "
            f"Root cause: **{root_cause}**. "
            f"Resolution: {fix_preview}. "
            "This report was generated via the KB fast-path without deploying sub-agents."
        )
    return (
        f"At {incident_time[:19].replace('T', ' ')} UTC, CloudWatch alarm `{error_code}` fired. "
        f"Automated sub-agents investigated logs, metrics, and deployments. "
        f"Root cause identified: **{root_cause}**. "
        f"Primary recommended fix: {fix_preview}."
    )
