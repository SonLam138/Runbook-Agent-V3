import json
import os
from datetime import datetime

LOG_FILE = r"D:\agent_logging\agent.log"


def ensure_log_dir():
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    except Exception:
        pass


def log_event(event, session_id, data=None, decision_trace=None):
    try:
        ensure_log_dir()

        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "session_id": session_id,

            # --- flatten fields for dashboard ---
            "action": decision_trace.get("action") if decision_trace else None,
            "confidence": decision_trace.get("confidence") if decision_trace else None,

            # --- payload ---
            "data": data or {},

            # --- reasoning ---
            "decision_trace": decision_trace or {}
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    except Exception:
        # 🔥 NEVER break agent
        pass

# ============================================================
# V4 LOGGING - CLARIFY SERVICE STEP
# ============================================================

def log_service_resolve_attempt(
    session_id,
    state,
    user_input,
    resolved_service=None,
    confidence=None,
    result="fail",
    reason=None,
):
    log_event(
        event="service_resolve_attempt",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "mode": state.get("mode"),
            "pending_slot": state.get("pending_slot"),
            "user_input": user_input,
            "resolved_service": resolved_service,
            "confidence": confidence,
            "result": result,
            "reason": reason,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "service_resolve",
            "reason": reason,
            "confidence": confidence if confidence else 0.0
        }
    )


def log_service_clarify_start(
    session_id,
    state,
    reason="cannot_resolve_service",
):
    log_event(
        event="service_clarify_start",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "mode": state.get("mode"),
            "pending_slot": state.get("pending_slot"),
            "reason": reason,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "clarify_start",
            "reason": reason,
            "confidence": 0.0
        }
    )


def log_service_clarify_input(
    session_id,
    state,
    user_input,
):
    log_event(
        event="clarify_input",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "slot": "service",
            "mode": state.get("mode"),
            "pending_slot": state.get("pending_slot"),
            "user_input": user_input,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "receive_input",
            "reason": "user_response_in_clarify",
            "confidence": 1.0
        }
    )


def log_service_clarify_extract(
    session_id,
    state,
    user_input,
    extracted_service=None,
    confidence=None,
    result="fail",
    reason=None,
):
    log_event(
        event="clarify_extract",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "slot": "service",
            "mode": state.get("mode"),
            "pending_slot": state.get("pending_slot"),
            "user_input": user_input,
            "extracted_value": extracted_service,
            "confidence": confidence,
            "result": result,
            "reason": reason,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "extract_service",
            "reason": reason,
            "confidence": confidence if confidence else 0.0
        }
    )


def log_service_clarify_update(
    session_id,
    state,
    old_service,
    new_service,
    next_slot="issue_type",
    reason="service_collected",
):
    log_event(
        event="clarify_update",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "slot": "service",
            "mode": state.get("mode"),
            "pending_slot": state.get("pending_slot"),
            "old_value": old_service,
            "new_value": new_service,
            "next_slot": next_slot,
            "reason": reason,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "update_service",
            "reason": reason,
            "confidence": 1.0
        }
    )


def log_service_clarify_retry(
    session_id,
    state,
    user_input,
    reason="cannot_extract_service",
):
    log_event(
        event="clarify_retry",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "slot": "service",
            "mode": state.get("mode"),
            "pending_slot": state.get("pending_slot"),
            "user_input": user_input,
            "reason": reason,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "retry_clarify",
            "reason": reason,
            "confidence": 0.0
        }
    )

# ============================================================
# V4 LOGGING - ISSUE TYPE STEP
# ============================================================

def log_issue_type_resolve_attempt(
    session_id,
    state,
    user_input,
    service,
    result
):
    log_event(
        event="issue_type_resolve_attempt",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "service": service,
            "user_input": user_input,
            "result": result,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "resolve_issue_type",
            "reason": "analyze_user_input",
            "confidence": 1.0 if result.get("issue_type") else 0.0
        }
    )


def log_issue_type_selected(
    session_id,
    state,
    service,
    issue_type,
    source="resolve"
):
    log_event(
        event="issue_type_selected",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "service": service,
            "issue_type": issue_type,
            "source": source,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "select_issue_type",
            "reason": "matched_from_input",
            "confidence": 1.0
        }
    )


def log_clarify_query_built(
    session_id,
    state,
    query,
    service,
    issue_type
):
    log_event(
        event="clarify_query_built",
        session_id=session_id,
        data={
            "flow_context": "clarify_search",
            "query": query,
            "service": service,
            "issue_type": issue_type
        },
        decision_trace={
            "action": "build_query",
            "reason": "issue_type_collected",
            "confidence": 1.0
        }
    )


def log_clarify_strong_match(
    session_id,
    state,
    query,
    selected_rb
):
    log_event(
        event="clarify_strong_match",
        session_id=session_id,
        data={
            "flow_context": "clarify_search",
            "query": query,
            "selected_title": selected_rb.get("title"),
            "score": selected_rb.get("score", 0.0),
        },
        decision_trace={
            "action": "return_runbook",
            "reason": "strong_match_after_error_message",
            "confidence": selected_rb.get("score", 0.0)
        }
    )


def log_clarify_next_issue_type(
    session_id,
    state,
    from_slot=None,
    to_slot=None,
    reason=None

):
    log_event(
        event="clarify_next",
        session_id=session_id,
        data={
            "flow_context": "clarify",
            "from_slot": from_slot,
            "to_slot": to_slot,
            "reason": reason,
            "collected_slots": state.get("slots", {}).copy(),
        },
        decision_trace={
            "action": "continue_clarify",
            "reason": reason,
            "confidence": 0.0
        }
    )