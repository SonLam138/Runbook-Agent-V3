import json
from pathlib import Path
from copy import deepcopy
from app.semantic_cache import rebuild_runtime_semantic_cache_from_sessions

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SESSION_FILE = BASE_DIR / "data" / "sessions.json"

SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

# =====================================================
# ✅ RUNTIME SESSION (QUAN TRỌNG NHẤT CHO PHASE 6)
# =====================================================
# 👉 Giữ state trong RAM giữa các lần gọi agent
RUNTIME_SESSIONS = {}

# =====================================================
# DEFAULT STATE
# =====================================================
DEFAULT_STATE = {
    # ===== core state =====
    "mode": "idle",
    "original_query": "",
    "clarify_turns": 0,
    "pending_slot": None,

    # ===== structured memory =====
    "slots": {
        "issue_type": None,
        "service": None,
        "error_message": None
    },

    # ===== conversation =====
    "history": [],

    # ===== semantic memory =====
    "semantic_query": "",

    # ===== failure handling =====
    "last_runbook": None,
    "tried_runbooks": [],
    "last_result_status": None,

    # ===== semantic cache (per session) =====
    "semantic_cache": [],

    # ===== ✅ CANDIDATE CONTEXT (PHASE 6)
    "candidate_context": {
        "active": False,
        "original_query": None,
        "refinement_history": [],
        "tier1": [],
        "tier2_lanes": [],
        "current_lane_candidates": [],
        "current_lane": "tier1",
        "current_lane_label": "Hướng chính",
        "current_index": 0,
        "service_locked": False,
        "resolved_service": None,
        "service_candidates": [],
        "clarify_reason": None,
        "flow_state": None,
        "show_full_runbook": False
    },

    # ===== debug =====
    "last_action": None
}

# =====================================================
# LOAD / SAVE FILE
# =====================================================
def load_sessions():
    if not SESSION_FILE.exists():
        return {}

    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)

        # rebuild semantic cache
        rebuild_runtime_semantic_cache_from_sessions(sessions)

        return sessions

    except Exception as e:
        print("⚠️ load_sessions error:", e)
        return {}


def save_sessions(data):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ save_sessions error:", e)


# =====================================================
# ✅ SESSION API (FIX CHÍNH Ở ĐÂY)
# =====================================================
def get_session(session_id: str) -> dict:

    # ✅ PRIORITY 1: runtime cache (KHÔNG reset giữa calls)
    if session_id in RUNTIME_SESSIONS:
        return RUNTIME_SESSIONS[session_id]

    # ✅ fallback: load từ file
    sessions = load_sessions()

    if session_id not in sessions:
        sessions[session_id] = deepcopy(DEFAULT_STATE)
        save_sessions(sessions)

    # ✅ cache vào RAM
    RUNTIME_SESSIONS[session_id] = sessions[session_id]

    return RUNTIME_SESSIONS[session_id]


def update_session(session_id: str, state: dict):

    # ✅ update runtime trước
    RUNTIME_SESSIONS[session_id] = state

    # ✅ sync xuống file
    sessions = load_sessions()
    sessions[session_id] = state
    save_sessions(sessions)


def reset_session(session_id: str):

    state = deepcopy(DEFAULT_STATE)

    # ✅ reset runtime
    RUNTIME_SESSIONS[session_id] = state

    # ✅ reset file
    sessions = load_sessions()
    sessions[session_id] = state
    save_sessions(sessions)