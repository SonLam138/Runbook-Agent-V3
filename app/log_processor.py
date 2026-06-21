# ============================================================
# V4 SECRETARY - LOG PROCESSOR V2 (PRODUCTION READY)
# ============================================================

import json
from pathlib import Path


DEFAULT_RAW_LOG_PATH = Path(r"D:\agent_logging\agent.log")
DEFAULT_OUTPUT_PATH = Path(r"D:\Runbook_Agent\datasets\v4_training_dataset.jsonl")


# ============================================================
# LOAD LOG
# ============================================================
def load_raw_logs():
    logs = []

    with open(DEFAULT_RAW_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # handle prefix logs
            if "{" in line:
                line = line[line.index("{"):]

            try:
                logs.append(json.loads(line))
            except Exception as e:
                print("⚠️ Skip line:", e)

    return logs


# ============================================================
# GROUP SESSION
# ============================================================
def group_by_session(logs):
    sessions = {}

    for log in logs:
        sid = log.get("session_id")
        if not sid:
            continue

        sessions.setdefault(sid, []).append(log)

    return sessions


# ============================================================
# BUILD STEPS
# ============================================================
def build_steps(session_logs, session_id):
    steps = []
    current = None
    turn = 0

    latest_slots = {
        "service": None,
        "issue_type": None,
        "error_message": None
    }

    for log in session_logs:
        event = log.get("event")
        data = log.get("data", {}) or {}
        decision = log.get("decision_trace", {}) or {}

        # =========================
        # NEW TURN
        # =========================
        if event == "agent_start":
            if current:
                finalize(current)
                steps.append(current)

            turn += 1

            current = {
                "session_id": session_id,
                "turn_index": turn,
                "user_input": data.get("user_input"),
                "mode": None,
                "pending_slot": None,
                "slots": latest_slots.copy(),
                "semantic_query": None,
                "retrieval": None,
                "decision": None,
                "output": None,
                "events": []
            }

        if current is None:
            continue

        current["events"].append(event)

        # =========================
        # SLOT UPDATE
        # =========================
        slots = data.get("collected_slots")
        if isinstance(slots, dict):
            for k, v in slots.items():
                if v is not None:
                    latest_slots[k] = v

        # 🔥 FIX: infer error_message directly
        if event == "clarify_input":
            slot = data.get("slot")
            val = data.get("user_input")

            if slot == "error_message" and val:
                latest_slots["error_message"] = val

        current["slots"] = latest_slots.copy()

        # =========================
        # MODE + SLOT INFER
        # =========================
        if data.get("mode"):
            current["mode"] = data["mode"]

        if data.get("pending_slot"):
            current["pending_slot"] = data["pending_slot"]

        # 🔥 infer from clarify_input
        if event == "clarify_input":
            current["mode"] = "clarifying"
            if data.get("slot"):
                current["pending_slot"] = data["slot"]

        # =========================
        # QUERY
        # =========================
        if data.get("semantic_query"):
            current["semantic_query"] = data["semantic_query"]

        if data.get("query"):
            current["semantic_query"] = current["semantic_query"] or data["query"]

        if event == "clarify_query_built":
            current["semantic_query"] = data.get("query")

        # =========================
        # RETRIEVAL
        # =========================
        if event == "retrieval_done":
            current["retrieval"] = {
                "query": data.get("query"),
                "top_k": data.get("top_k"),
                "candidate_count": data.get("candidate_count")
            }

        # =========================
        # DECISION
        # =========================
        if decision:
            current["decision"] = {
                "action": decision.get("action"),
                "reason": decision.get("reason"),
                "confidence": decision.get("confidence")
            }

        # =========================
        # OUTPUT
        # =========================

        # ✅ strong match
        if event in ("strong_match_found", "clarify_strong_match"):
            current["output"] = {
                "type": "runbook",
                "title": data.get("selected_title"),
                "score": data.get("score")
            }

        # ✅ clarify next
        if event == "clarify_next":
            current["output"] = {
                "type": "clarify",
                "from_slot": data.get("from_slot"),
                "to_slot": data.get("to_slot")
            }

        # 🔥 FIX: clarify_update → next slot
        if event == "clarify_update":
            current["output"] = {
                "type": "clarify",
                "from_slot": data.get("slot"),
                "to_slot": data.get("next_slot")
            }

    if current:
        finalize(current)
        steps.append(current)

    return steps


# ============================================================
# FINALIZE STEP
# ============================================================
def finalize(step):
    events = step.get("events", [])
    decision = step.get("decision", {})
    retrieval = step.get("retrieval")

    # =========================
    # OUTCOME
    # =========================
    if "clarify_strong_match" in events or decision.get("action") == "return_runbook":
        step["outcome"] = "success"

    elif "service_clarify_start" in events:
        step["outcome"] = "needs_clarification"

    elif "clarify_next" in events:
        step["outcome"] = "needs_clarification"

    elif "clarify_update" in events:
        step["outcome"] = "continue_clarification"
        
    elif "clarify_stop" in events:
        step["outcome"] = "stopped"

    elif retrieval:
        step["outcome"] = "retrieved_no_decision"

    else:
        step["outcome"] = "unknown"

    # =========================
    # LABEL
    # =========================
    if "clarify_strong_match" in events:
        step["label"] = "positive_match"

    elif "service_clarify_start" in events:
        step["label"] = "needs_service_clarification"

    elif "clarify_next" in events:
        step["label"] = "needs_more_info"

    elif "clarify_update" in events:
        step["label"] = "slot_filled_continue"

    elif retrieval:
        step["label"] = "retrieved_no_strong_match"

    else:
        step["label"] = "unknown"


# ============================================================
# BUILD DATASET
# ============================================================
def build_dataset(raw_logs):
    sessions = group_by_session(raw_logs)
    dataset = []

    for sid, logs in sessions.items():
        steps = build_steps(logs, sid)
        dataset.extend(steps)

    return dataset


# ============================================================
# SAVE
# ============================================================
def save_dataset(data):
    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✅ Dataset saved:", DEFAULT_OUTPUT_PATH)
    print("✅ Total records:", len(data))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logs = load_raw_logs()
    dataset = build_dataset(logs)
    save_dataset(dataset)