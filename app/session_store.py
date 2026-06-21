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
# DEFAULT STATE (RẤT QUAN TRỌNG)
# =====================================================
DEFAULT_STATE = {
    # ===== core state =====
    "mode": "idle",                  # idle | clarifying
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

    # ===== debug =====
    "last_action": None
}


# =====================================================
# LOAD / SAVE
# =====================================================
def load_sessions():
    if not SESSION_FILE.exists():
        return {}

    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)

        # 🔥 NEW: rebuild semantic cache runtime
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
# SESSION API
# =====================================================
def get_session(session_id: str) -> dict:
    sessions = load_sessions()

    if session_id not in sessions:
        sessions[session_id] = deepcopy(DEFAULT_STATE)
        save_sessions(sessions)

    return sessions[session_id]


def update_session(session_id: str, state: dict):
    sessions = load_sessions()
    sessions[session_id] = state
    save_sessions(sessions)


def reset_session(session_id: str):
    sessions = load_sessions()
    sessions[session_id] = deepcopy(DEFAULT_STATE)
    save_sessions(sessions)