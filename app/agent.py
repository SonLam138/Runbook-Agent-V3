import json
from pathlib import Path
import re
import numpy as np
import faiss
#from app.vector_store import FaissStore
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.retrieval_multiquery import retrieve_candidates_multiquery
from app.semantic_cache import load_runbook_data, RUNBOOK_DATA
from app.service_resolver_v2 import resolve_service_v2
from app.llm import call_llm
from app.session_store import get_session, update_session, reset_session
#from app.tool import retrieve_candidates_meta, search_RB_topk
from app.vector_store import ChromaStore
from app.load_model import embedding_model as CACHE_MODEL
from app.load_knowledge import get_clarify_knowledge, get_decision_knowledge
from app.issue_type_resolver import resolve_issue_type
from Log.logger import (log_event, log_service_resolve_attempt, log_service_clarify_update, log_service_clarify_extract, log_service_clarify_input, log_service_clarify_retry, log_service_clarify_start)
from app.semantic_cache import (enrich_runbook_from_json,check_semantic_cache,remember_success)
from Log.logger import (log_issue_type_resolve_attempt, log_issue_type_selected, log_clarify_next_issue_type, log_clarify_query_built, log_clarify_strong_match)
# =====================================================
# CONFIG
# =====================================================
MAX_CLARIFY_TURNS = 5
MAX_RETRY_RUNBOOKS = 3
CACHE_THRESHOLD = 0.82
# Nếu top score thấp hơn ngưỡng này → KHÔNG được trả runbook
MIN_CANDIDATE_CONFIDENCE = 0.59

# Nếu top score cao hơn ngưỡng này → search luôn, không cần LLM
STRONG_MATCH_THRESHOLD = 0.65

FAILURE_SIGNALS = [
    "vẫn bị lỗi",
    "vẫn lỗi",
    "không được",
    "vẫn không được",
    "làm rồi vẫn lỗi",
    "vẫn không login được",
    "vẫn không vào được",
    "chưa được",
    "không ổn",
    "vẫn fail",
    "chưa ổn",
]

NEGATIVE_HINTS = [
    "không",
    "chưa",
    "vẫn",
    "lỗi",
    "sai",
    "fail",
    "not",
    "error",
    "unable"
]

FLOW_FAST_MATCH = "fast_match"
FLOW_OPEN_RUNBOOK = "open_runbook"
FLOW_NEXT_PRIMARY = "next_primary"

FLOW_SERVICE_LOCK_PROMPT = "service_lock_prompt"
FLOW_SERVICE_LOCKED = "service_locked"
FLOW_ADVISORY_PRE_CANDIDATE = "advisory_pre_candidate"
FLOW_ADVISORY_TIER2 = "advisory_tier2"
MAX_ISSUE_REFINE_ATTEMPTS = 4
FLOW_CONFIRM_SERVICE_SWITCH = "confirm_service_switch"



load_runbook_data()
# =====================================================
# BASIC HELPERS
# =====================================================
def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_parse_json(text):
    """
    Parse JSON safely from LLM output.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("⚠️ JSON parse error:", e)
    return None


def ensure_state(state):
    """
    Session-based conversation state.
    """
    if "candidate_context" not in state:
        state["candidate_context"] = {
            "active": False,
            "original_query": None,
            "refinement_history": [],
            "tier1": [],
            "tier2_lanes": [],
            "current_lane_candidates": [],
            "current_lane": "tier1",
            "current_lane_label": "Hướng chính",
            "current_index": 0,
            "show_full_runbook": False

        }
    state.setdefault("mode", "idle")                  # idle | clarifying
    state.setdefault("original_query", "")
    state.setdefault("clarify_turns", 0)
    state.setdefault("pending_slot", None)            # issue_type | service | error_message

    state.setdefault("slots", {
        "issue_type": None,
        "service": None,
        "error_message": None
    })

    state.setdefault("history", [])

    # semantic memory
    state.setdefault("semantic_query", "")

    # retry / result memory
    state.setdefault("last_runbook", None)
    state.setdefault("tried_runbooks", [])
    state.setdefault("last_result_status", None)
    state.setdefault("last_action", None)

    # semantic cache trace per session
    state.setdefault("semantic_cache", [])


def reply(session_id, state, message):
    """
    Standardized output:
    - append assistant message
    - persist session
    """
    message = message or "❌ Hệ thống chưa tạo được phản hồi. Bạn vui lòng thử lại."
    state["history"].append({"role": "assistant", "text": message})
    update_session(session_id, state)
    return message


def start_clarify(state, user_input, next_slot):
    """
    Đưa agent vào clarify mode một cách chuẩn.
    """
    state["mode"] = "clarifying"
    state["pending_slot"] = next_slot or "issue_type"
    state["last_action"] = "ask_more"

    if not state["original_query"]:
        state["original_query"] = user_input

    if state["clarify_turns"] == 0:
        state["clarify_turns"] = 1

# =====================================================
# PROMPT LOADER
# =====================================================

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(prompt_file_name: str) -> str:
    """
    Load prompt template từ thư mục /prompts.
    agent.py nằm trong /app nên parent.parent sẽ trỏ về root project.
    """
    prompt_path = PROMPT_DIR / prompt_file_name

    if not prompt_path.exists():
        print(f"⚠️ Prompt file not found: {prompt_path}")
        return ""

    return prompt_path.read_text(encoding="utf-8")

def render_prompt(template: str, variables: dict) -> str:
    """
    Replace các biến dạng {{VAR_NAME}} trong prompt template.
    Không dùng format() để tránh lỗi khi prompt có dấu ngoặc nhọn.
    """
    rendered = template

    for key, value in variables.items():
        if value is None:
            value = "Chưa có"

        if isinstance(value, (list, dict)):
            value = str(value)

        rendered = rendered.replace("{{" + key + "}}", str(value))

    return rendered

def build_clarify_knowledge_context(state, service=None, issue_type=None, error_message=None):
    """
    Build knowledge context ngắn cho LLM.
    Ưu tiên dùng get_clarify_knowledge(service) hiện có của bạn.
    Sau này có thể enrich thêm từ keyword_catalog.json hoặc candidate runbooks.
    """

    context_lines = []

    if service:
        knowledge = get_clarify_knowledge(service)
    else:
        knowledge = ""

    if knowledge:
        context_lines.append("Knowledge grounding theo service:")
        context_lines.append(str(knowledge))

    context_lines.append("")
    context_lines.append("Slot/context hiện tại:")
    context_lines.append(f"- Service: {service or 'Chưa có'}")
    context_lines.append(f"- Issue type: {issue_type or 'Chưa có'}")
    context_lines.append(f"- Error message: {error_message or 'Chưa có'}")
    context_lines.append(f"- Semantic query: {state.get('semantic_query') or 'Chưa có'}")

    # Nếu sau này bạn lưu candidate gần nhất vào state thì tự dùng luôn
    last_candidates = state.get("last_candidates") or state.get("last_candidate_titles")
    if last_candidates:
        context_lines.append(f"- Candidate runbook gần nhất: {last_candidates}")

    # Keyword gợi ý fallback theo service
    service_lower = normalize(service or "")

    service_keyword_map = {
        "exchange": [
            "mailbox",
            "gửi mail",
            "nhận mail",
            "dung lượng mailbox",
            "phân quyền mailbox",
            "đăng nhập mailbox",
            "outlook",
        ],
        "active directory": [
            "đăng nhập",
            "reset password",
            "khóa tài khoản",
            "unlock account",
            "phân quyền",
            "group policy",
            "user account",
        ],
        "vpn": [
            "kết nối VPN",
            "xác thực MFA",
            "timeout",
            "không kết nối được",
            "sai tài khoản mật khẩu",
            "mất kết nối",
        ],
    }

    matched_keywords = []

    for svc, keywords in service_keyword_map.items():
        if svc in service_lower:
            matched_keywords.extend(keywords)

    if matched_keywords:
        context_lines.append("")
        context_lines.append("Keyword CNTT liên quan service:")
        context_lines.append(", ".join(matched_keywords))
    else:
        context_lines.append("")
        context_lines.append("Keyword CNTT gợi ý chung:")
        context_lines.append(
            "đăng nhập, xác thực, tài khoản, phân quyền, kết nối, timeout, đồng bộ, dung lượng, gửi nhận, MFA"
        )

    return "\n".join(context_lines)

# ============================================================
# NORMALIZE SERVICE RESOLVE_V2 RESULT
# ============================================================
def normalize_service_result(result):
    """
    Chuẩn hóa output từ service resolver.

    Với resolver_v2, result đã có:
    - service
    - confidence
    - source
    - is_strong
    - status
    - reason
    - candidates

    Agent chỉ normalize, KHÔNG tự quyết confidence nữa.
    """

    if result is None:
        return {
            "service": None,
            "confidence": None,
            "source": "none",
            "is_strong": False,
            "status": "unresolved",
            "reason": "resolver_returned_none",
            "candidates": [],
        }

    # Case legacy: resolver trả string
    if isinstance(result, str):
        service = result.strip()
        return {
            "service": service if service else None,
            "confidence": 1.0 if service else None,
            "source": "legacy_string",
            "is_strong": bool(service),
            "status": "resolved" if service else "unresolved",
            "reason": "legacy_string_result",
            "candidates": [],
        }

    # Case legacy: tuple/list
    if isinstance(result, (tuple, list)):
        service = result[0] if len(result) >= 1 else None
        confidence = result[1] if len(result) >= 2 else None

        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None

        return {
            "service": service.strip() if isinstance(service, str) and service.strip() else None,
            "confidence": confidence,
            "source": "legacy_tuple",
            "is_strong": confidence is not None and confidence >= 0.75,
            "status": "resolved" if service else "unresolved",
            "reason": "legacy_tuple_result",
            "candidates": [],
        }

    # Case resolver_v2 dict
    if isinstance(result, dict):
        service = result.get("service")
        confidence = result.get("confidence")

        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None

        return {
            "service": service.strip() if isinstance(service, str) and service.strip() else None,
            "confidence": confidence,
            "source": result.get("source", "unknown"),
            "is_strong": bool(result.get("is_strong", False)),
            "status": result.get("status", "unresolved"),
            "reason": result.get("reason", ""),
            "candidates": result.get("candidates", []),
        }

    return {
        "service": None,
        "confidence": None,
        "source": "unknown",
        "is_strong": False,
        "status": "unresolved",
        "reason": "unsupported_result_type",
        "candidates": [],
    }

def normalize_service_detection(raw_result):
    """
    ✅ Chuẩn hoá output từ service_resolver
    Trả về format chuẩn cho Phase 8A:

    {
        "resolved_service": str | None,
        "candidate_services": list[str],
        "confidence": float,
        "is_strong": bool
    }
    """

    # =============================
    # ✅ EMPTY / NONE GUARD
    # =============================
    if not raw_result or not isinstance(raw_result, dict):
        return {
            "resolved_service": None,
            "candidate_services": [],
            "confidence": 0.0,
            "is_strong": False
        }

    # =============================
    # ✅ RESOLVED SERVICE
    # =============================
    resolved_service = (
        raw_result.get("service")
        or raw_result.get("resolved_service")
        or raw_result.get("service_name")
    )

    # =============================
    # ✅ RAW CANDIDATES
    # =============================
    raw_candidates = (
        raw_result.get("candidate_services")
        or raw_result.get("candidates")
        or raw_result.get("services")
        or []
    )

    # =============================
    # ✅ CLEAN CANDIDATES
    # =============================
    candidate_services = []

    for c in raw_candidates:
        if isinstance(c, dict):
            name = (
                c.get("service")
                or c.get("service_name")
                or c.get("name")
            )
            if name:
                candidate_services.append(str(name))

        elif isinstance(c, str):
            candidate_services.append(c)

    # ✅ nếu có resolved_service mà chưa có trong list → add vào
    if resolved_service:
        resolved_service = str(resolved_service)
        if resolved_service not in candidate_services:
            candidate_services.insert(0, resolved_service)

    # =============================
    # ✅ CONFIDENCE
    # =============================
    conf = raw_result.get("confidence", 0.0)

    try:
        confidence = float(conf)
    except Exception:
        confidence = 0.0

    # =============================
    # ✅ IS STRONG (RULE CHUẨN)
    # =============================
    is_strong = False

    if resolved_service and confidence >= 0.7:
        is_strong = True

    # =============================
    # ✅ FINAL OUTPUT
    # =============================
    return {
        "resolved_service": resolved_service,
        "candidate_services": candidate_services,
        "confidence": confidence,
        "is_strong": is_strong
    }

# =====================================================
# OUTPUT FORMAT
# =====================================================
def format_runbook(rb, prefix=None):
    if not rb:
        return "❌ Không tìm thấy runbook phù hợp."

    precheck = "\n- ".join(rb.get("precheck", [])) or "(không có)"
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(rb.get("steps", []))) or "(không có)"
    postcheck = "\n- ".join(rb.get("postcheck", [])) or "(không có)"

    blocks = []
    if prefix:
        blocks.append(prefix)

    blocks.append(f"""
Bạn vui lòng thử làm theo hướng dẫn sau:

📘 Tiêu đề: {rb.get('title','')}

🖥 Dịch vụ: {rb.get('service','')}

🔹 Điều kiện thực hiện:
- {precheck}

🔹 Các bước thực hiện:
{steps}

🔹 Kiểm tra sau thực hiện:
- {postcheck}
""".strip())

    return "\n\n".join(blocks)


# =====================================================
# SLOT FILLING / CLARIFY SUPPORT
# =====================================================
def fill_pending_slot(state, user_input):
    pending = state.get("pending_slot")
    if pending in state["slots"]:
        state["slots"][pending] = user_input.strip()


def is_generic_clarify_message(msg: str):
    """
    Nhận diện các câu hỏi quá chung chung, không giúp fill slot thật sự.
    """
    if not msg:
        return True

    m = normalize(msg)

    generic_patterns = [
        "bạn có thể mô tả rõ hơn",
        "mô tả rõ hơn",
        "mô tả chi tiết hơn",
        "cho tôi biết thêm",
        "vấn đề bạn đang gặp",
        "bạn đang gặp vấn đề gì",
    ]

    return any(p in m for p in generic_patterns)

def generate_clarify_message_llm(user_input, state, next_slot):
    """
    Dùng LLM để sinh câu hỏi clarify mềm mại theo slot còn thiếu.

    Version mới:
    - Prompt tách ra file riêng trong /prompts.
    - LLM bị khóa trong context CNTT + slot state + knowledge grounding.
    - Không cho LLM hỏi lại thông tin đã biết.
    - Không để LLM tự quyết định flow.
    """

    # =====================================================
    # 1) Lấy context từ state
    # =====================================================

    slots = state.get("slots", {})

    resolved_service = state.get("resolved_service")
    known_service = slots.get("service") or resolved_service
    known_issue = slots.get("issue_type")
    known_error = slots.get("error_message")
    semantic_query = state.get("semantic_query")

    # =====================================================
    # 2) Load prompt template từ file
    # =====================================================

    template = load_prompt_template("clarify_error_message_prompt.txt")

    if not template:
        print("⚠️ Empty clarify prompt template")
        return None

    # =====================================================
    # 3) Build knowledge context
    # =====================================================

    knowledge_context = build_clarify_knowledge_context(
        state=state,
        service=known_service,
        issue_type=known_issue,
        error_message=known_error,
    )

    # =====================================================
    # 4) Render prompt
    # =====================================================

    prompt = render_prompt(
        template,
        {
            "MODE": state.get("mode"),
            "PENDING_SLOT": next_slot,
            "SERVICE": known_service,
            "ISSUE_TYPE": known_issue,
            "ERROR_MESSAGE": known_error,
            "RESOLVED_SERVICE": resolved_service,
            "SEMANTIC_QUERY": semantic_query,
            "USER_INPUT": user_input,
            "KNOWLEDGE_CONTEXT": knowledge_context,
        }
    )

    # =====================================================
    # 5) Gọi LLM
    # =====================================================

    try:
        msg = call_llm(prompt)
        msg = (msg or "").strip()

        if not msg:
            return None

        # Chỉ lấy dòng đầu tiên để tránh LLM giải thích dài
        msg = msg.split("\n")[0].strip()

        # Bỏ dấu ngoặc kép nếu LLM tự thêm
        msg = msg.strip('"').strip("'").strip()

        # =====================================================
        # 6) Guard output
        # =====================================================

        msg_lower = normalize(msg)
        user_text = normalize(user_input)

        user_said_no_error_code = (
            "không" in user_text
            and (
                "mã lỗi" in user_text
                or "ma loi" in user_text
                or "error" in user_text
                or "thông báo lỗi" in user_text
                or "thong bao loi" in user_text
            )
        )

        # Guard 1: generic clarify message
        if is_generic_clarify_message(msg):
            print("⚠️ Clarify rejected as generic:", msg)
            return None

        # Guard 2: đã biết service thì không được hỏi lại service
        if known_service and (
            "dịch vụ nào" in msg_lower
            or "dich vu nao" in msg_lower
            or "hệ thống nào" in msg_lower
            or "he thong nao" in msg_lower
            or "đang thao tác trên hệ thống" in msg_lower
            or "dang thao tac tren he thong" in msg_lower
        ):
            print("⚠️ Clarify rejected because it asks known service again:", msg)

            return (
                f"Với {known_service}, lỗi xảy ra ở thao tác hoặc chức năng nào, "
                "ví dụ đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
            )

        # Guard 3: nếu user đã nói không có mã lỗi thì không hỏi lại mã lỗi
        if user_said_no_error_code and (
            "mã lỗi" in msg_lower
            or "ma loi" in msg_lower
            or "error code" in msg_lower
            or "thông báo lỗi" in msg_lower
            or "thong bao loi" in msg_lower
        ):
            print("⚠️ Clarify rejected because user already said no error code:", msg)

            if known_service:
                return (
                    f"Với {known_service}, lỗi xảy ra ở thao tác hoặc chức năng nào, "
                    "ví dụ đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
                )

            return (
                "Lỗi xảy ra ở thao tác hoặc chức năng CNTT nào, "
                "ví dụ đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
            )

        # Guard 4: không cho câu quá rộng dù chưa bị generic detector bắt
        too_generic_patterns = [
            "bạn gặp lỗi gì",
            "ban gap loi gi",
            "bạn muốn gì",
            "ban muon gi",
            "nói rõ hơn",
            "noi ro hon",
            "cung cấp thêm thông tin",
            "cung cap them thong tin",
            "mô tả rõ hơn",
            "mo ta ro hon",
        ]

        if any(pattern in msg_lower for pattern in too_generic_patterns):
            print("⚠️ Clarify rejected as too broad:", msg)

            if known_service:
                return (
                    f"Với {known_service}, lỗi đang xảy ra ở thao tác/chức năng nào "
                    "như đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
                )

            return (
                "Lỗi đang xảy ra ở thao tác/chức năng CNTT nào "
                "như đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
            )

        return msg

    except Exception as e:
        print("⚠️ Clarify LLM error:", e)

    return None

def generate_clarify_message_fallback(next_slot, user_input):
    if next_slot == "issue_type":
        return "Bạn đang gặp lỗi gì cụ thể? Ví dụ: không đăng nhập được, access denied hay lỗi khác?"
    elif next_slot == "service":
        return "Bạn đang thao tác trên hệ thống hoặc dịch vụ nào? Ví dụ: AD, Exchange, Windows?"
    elif next_slot == "error_message":
        return "Bạn có thông báo lỗi hoặc mã lỗi cụ thể nào không?"
    else:
        return f"Bạn mô tả rõ hơn về '{user_input}' giúp tôi nhé?"


def build_semantic_query(state):
    """
    Reformulate query from original_query + filled slots.
    Use LLM first; fallback to simple join if needed.
    """
    slots = state["slots"]

    history_text = "\n".join(
        f"{item['role']}: {item['text']}" for item in state["history"]
    )

    prompt = f"""
Bạn là IT agent.

Nhiệm vụ:
Từ lịch sử hội thoại và thông tin đã biết, hãy viết lại thành 1 câu query NGẮN GỌN, bằng tiếng Việt,
dùng để tìm runbook phù hợp nhất.

QUAN TRỌNG:
- Chỉ trả về 1 dòng text
- KHÔNG giải thích
- KHÔNG dịch sang tiếng Anh

Original query:
{state["original_query"]}

Known slots:
- issue_type: {slots.get("issue_type")}
- service: {slots.get("service")}
- error_message: {slots.get("error_message")}

History:
{history_text}
"""

    raw = call_llm(prompt)
    semantic_query = (raw or "").strip()

    if not semantic_query:
        parts = []

        if state["original_query"]:
            parts.append(state["original_query"])

        for key in ["issue_type", "service", "error_message"]:
            val = slots.get(key)
            if val:
                parts.append(val)

        semantic_query = " ".join(parts)

    state["semantic_query"] = semantic_query
    return semantic_query

# =====================================================
# FAILURE DETECTION (HYBRID)
# =====================================================
def rule_detect_failure(user_input):
    q = normalize(user_input)
    return any(sig in q for sig in FAILURE_SIGNALS)


def has_negative_hint(user_input):
    q = normalize(user_input)
    return any(h in q for h in NEGATIVE_HINTS)


def llm_detect_failure(user_input, state):
    if not state.get("last_runbook"):
        return False

    if state.get("last_action") != "search":
        return False

    if state.get("mode") != "idle":
        return False

    if state.get("last_result_status") not in ["returned", "retry_returned"]:
        return False

    if not has_negative_hint(user_input):
        return False

    prompt = f"""
Bạn là AI agent.

Ngữ cảnh:
- Agent đã cung cấp runbook cho user trước đó.
- Chỉ coi là FAILURE nếu user THỂ HIỆN RÕ rằng giải pháp trước không hiệu quả.

Last runbook:
- title: {state.get("last_runbook", {}).get("title")}
- service: {state.get("last_runbook", {}).get("service")}

User nói:
"{user_input}"

Câu hỏi:
User có đang PHẢN HỒI THẤT BẠI về runbook trước đó không?

Chỉ trả:
YES hoặc NO
"""

    try:
        raw = call_llm(prompt)
        result = (raw or "").strip().upper()
        print("🧠 RAW failure detection:", result)
        return "YES" in result
    except Exception as e:
        print("⚠️ LLM failure detection error:", e)
        return False


def is_failure(user_input, state):
    """
    Chỉ coi là failure/retry nếu trước đó agent đã trả runbook.
    Tránh việc query đầu tiên kiểu 'không đăng nhập được mail'
    bị hiểu nhầm là user đang phản hồi thất bại.
    """

    if not state.get("last_runbook"):
        return False

    if state.get("last_action") not in ["search", "retry"]:
        return False

    if state.get("mode") != "idle":
        return False

    if rule_detect_failure(user_input):
        print("✅ FAILURE detected by RULE")
        return True

    if not has_negative_hint(user_input):
        print("🟢 No negative hint → NOT FAILURE")
        return False

    if llm_detect_failure(user_input, state):
        print("✅ FAILURE detected by LLM")
        return True

    return False


# =====================================================
# CANDIDATE CONFIDENCE / STRONG MATCH
# =====================================================
def is_candidate_confident(full_candidates, min_score=MIN_CANDIDATE_CONFIDENCE):
    """
    If top candidate score is too low, do NOT allow search.
    """
    if not full_candidates:
        return False

    top_score = full_candidates[0].get("score", 0)
    print(f"🧪 CONFIDENCE CHECK: {top_score:.4f}")

    return top_score >= min_score


def detect_strong_match_by_score(full_candidates):
    """
    Strong match decision for semantic retrieval.

    Contract:
    - full_candidates đã được sort score giảm dần.
    - mỗi candidate có field "score".
    - return: (match: bool, idx: int | None)

    Logic:
    - Không chỉ dùng hard threshold.
    - Kết hợp best_score + margin giữa top1 và top2.
    """

    if not full_candidates:
        return False, None

    best_score = full_candidates[0].get("score", 0.0)

    # Nếu chỉ có 1 candidate, dùng ngưỡng an toàn cao hơn.
    if len(full_candidates) == 1:
        if best_score >= 0.60:
            return True, 0
        return False, None

    second_score = full_candidates[1].get("score", 0.0)
    margin = best_score - second_score

    print(
        f"🧠 STRONG MATCH CHECK: "
        f"best={best_score:.4f}, second={second_score:.4f}, margin={margin:.4f}"
    )

    # Case 1: match rõ
    if best_score >= 0.60 and margin >= 0.03:
        return True, 0

    # Case 2: borderline nhưng top1 cách top2 đủ xa
    if best_score >= 0.55 and margin >= 0.05:
        return True, 0

    return False, None


# =====================================================
# LLM DECISION ON CANDIDATES
# =====================================================
def decide_with_candidates(user_input, state, meta_candidates):
    service = state.get("resolved_service")
    knowledge = get_decision_knowledge(service)

    if not meta_candidates:
        return {
            "action": "ask_more",
            "selected_index": None,
            "message": "",
            "next_slot": "issue_type"
        }

    candidate_text = ""
    for c in meta_candidates:
        candidate_text += (
            f"[{c['idx']}] {c['title']} | {c['service']} | "
            f"{c['keyword']} | score={c['score']:.3f}\n"
        )

    raw = call_llm(f"""
{knowledge}

Bạn là IT agent nội bộ.

User query: "{user_input}"
Resolved service: {service}

Candidate runbook metadata:
{candidate_text}

NHIỆM VỤ:
- Nếu query match rõ 1 candidate => chọn "search"
- Nếu query còn mơ hồ => chọn "ask_more"

QUAN TRỌNG:
- Ưu tiên quyết định phù hợp với resolved service nếu đã có
- Chỉ chọn "search" khi thực sự có candidate phù hợp
- Nếu chưa chắc, hãy chọn "ask_more"
- Nếu ask_more thì KHÔNG cần message quá chi tiết ở đây, có thể để message rỗng

Trả JSON:
{{
  "action": "search hoặc ask_more",
  "selected_index": number hoặc null,
  "message": "...",
  "next_slot": "issue_type | service | error_message | null"
}}
""")

    parsed = safe_parse_json(raw)

    if parsed:
        return parsed

    return {
        "action": "ask_more",
        "selected_index": None,
        "message": "",
        "next_slot": "issue_type"
    }


# =====================================================
# FAILURE RETRY
# =====================================================
def handle_retry(session_id, state, vector_store):
    """
    Khi user phản hồi runbook trước chưa đúng:
    - dùng lại semantic_query gần nhất
    - loại bỏ các runbook đã trả trong tried_runbooks
    - trả runbook tiếp theo trong top-k
    """

    query = state.get("semantic_query")
    tried = state.get("tried_runbooks", [])

    print("🔁 RETRY REQUESTED")
    print("🔁 semantic_query:", query)
    print("🔁 tried_runbooks:", tried)

    if not query:
        return reply(
            session_id,
            state,
            "❌ Tôi chưa có đủ thông tin truy vấn trước đó để thử runbook khác. Bạn mô tả lại lỗi giúp tôi nhé."
        )

    results = vector_store.search(
        query=query,
        exclude_titles=tried,
        top_k=MAX_RETRY_RUNBOOKS
    )

    if not results:
        state["last_result_status"] = "retry_failed"
        state["last_action"] = "retry"

        return reply(
            session_id,
            state,
            "❌ Tôi chưa tìm thấy runbook khác phù hợp hơn với thông tin hiện tại. Bạn có thể bổ sung thêm thông báo lỗi hoặc mô tả chi tiết hơn không?"
        )

    rb = results[0]

    remember_success(state, query, rb)
    state["last_result_status"] = "retry_returned"
    state["last_action"] = "retry"
    state["mode"] = "idle"
    state["pending_slot"] = None

    return reply(
        session_id,
        state,
        "⚠️ Tôi sẽ thử runbook khác phù hợp hơn:\n\n" + format_runbook(rb)
    )
## Assistance Turn using LLM
def call_llm_companion(prompt, fallback):

    try:
        response = call_llm(prompt)   # ✅ dùng luôn wrapper của bạn

        if response and response.strip():
            return response.strip()

        return fallback

    except Exception as e:
        print("⚠️ LLM error:", e)
        return fallback


    
## Assistance Turn using LLM
def build_companion_prompt(ctx, turn_reason):

    primary = ctx.get("current_primary")
    current_lane = ctx.get("current_lane", "tier1")
    current_lane_label = ctx.get("current_lane_label", "Hướng chính")
    tier2_lanes = ctx.get("tier2_lanes", [])

    primary_label = primary.get("short_label") if primary else "chưa xác định"

    related_text = []
    for lane in tier2_lanes[:3]:
        lane_label = lane.get("lane_label")
        candidates = lane.get("candidates", [])
        first_label = candidates[0].get("short_label") if candidates else ""

        related_text.append(f"- {lane_label}: {first_label}")

    related_block = "\n".join(related_text) if related_text else "Không có hướng liên quan rõ."

    return f"""

Bạn là **IT Runbook Assistant**, vai trò của bạn là:
👉 đồng hành đang xử lý vấn đề khác, bạn có thể..."👉 đồng hành cùng người dùng
- "Bạn có thể xem runbook hoặc nói thêm..."

KHÔNG được giống:
- "Bước 1..."
- "Bạn cần làm..."
- "Thực hiện theo các bước sau"

Bây giờ hãy viết một câu phù hợp.

👉 giải thích nhẹ nhàng, ngắn gọn
👉 KHÔNG thực thi hệ thống

Ngữ cảnh:
- Runbook hiện tại: {primary_label}
- Lane hiện tại: {current_lane_label}

Các hướng liên quan:
{related_block}

# ❗ QUY TẮC CỰC KỲ QUAN TRỌNG

1. KHÔNG được:
- viết chi tiết các bước kỹ thuật
- liệt kê step runbook
- giả vờ đang thực hiện thao tác
- hướng dẫn từng bước như SOP

2. CHỈ ĐƯỢC:
- giải thích vì sao agent đề xuất hướng này dựa vào ngữ cảnh hiện tại
- gợi ý nhẹ nếu có khả năng lệch hướng
- mời user tiếp tục trao đổi

3. Văn phong:
- tối đa 2-3 câu
- tiếng Việt tự nhiên
- không dùng bullet list
- không dùng checklist

4. Luôn giữ vai trò:
👉 "người đồng hành", không phải "người thao tác"

5. Trong mọi câu trả lời luôn luôn dựa vào ngữ cảnh hiện tại, TUYỆT ĐỐI KHÔNG ĐƯỢC trả lời không theo ngữ cảnh đã chỉ định ở trên.

# ✅ OUTPUT MONG MUỐN

Ví dụ tốt:
- "Tôi đang nghiêng về hướng này vì..."

"""

## Assistance Turn using LLM
# def build_assistant_turn(ctx, turn_reason="candidate_presented"):

#     current_primary = ctx.get("current_primary")
#     tier2_lanes = ctx.get("tier2_lanes", [])

#     primary_label = (
#         current_primary.get("short_label")
#         if current_primary else "runbook hiện tại"
#     )

#     # =========================
#     # Fallback message theo từng tình huống
#     # =========================
#     if turn_reason == "candidate_presented":
#         fallback = (
#             f"Tôi đang nghiêng về hướng \"{primary_label}\". "
#             "Nếu đúng ngữ cảnh, bạn có thể mở runbook để xem chi tiết; "
#             "nếu chưa đúng, cứ nói thêm với tôi để tôi đổi hướng xử lý."
#         )

#     elif turn_reason == "open_runbook":
#         fallback = (
#             "Vâng, tôi sẽ mở nội dung runbook này. "
#             "Bạn cứ xem các bước bên dưới; nếu thấy chưa phù hợp, hãy nhắn lại cho tôi."
#         )

#     elif turn_reason == "next_primary":
#         fallback = (
#             f"Tôi chuyển sang một phương án khác trong cùng hướng. "
#             "Bạn xem thử runbook mới này có sát hơn không nhé."
#         )

#     elif turn_reason == "related_lane_selected":
#         fallback = (
#             "Đã hiểu, tôi sẽ chuyển sang hướng liên quan này. "
#             "Tôi sẽ mở đề xuất đầu tiên trong hướng đó để bạn kiểm tra."
#         )

#     else:
#         fallback = (
#             "Tôi sẽ tiếp tục đồng hành cùng bạn trong ngữ cảnh hiện tại. "
#             "Nếu chưa đúng hướng, bạn có thể mô tả thêm vấn đề."
#         )

#     prompt = build_companion_prompt(ctx, turn_reason)
#     message = call_llm_companion(prompt, fallback)

#     # =========================
#     # Suggested utterances
#     # Các action này là câu hội thoại, không phải lệnh kỹ thuật
#     # =========================
#     suggested = []

#     if current_primary:
#         suggested.append({
#             "label": "📘 Mở runbook này",
#             "utterance": "mở runbook này"
#         })

#     suggested.append({
#         "label": "➡️ Cho tôi phương án khác",
#         "utterance": "cho tôi phương án khác"
#     })

#     for lane in tier2_lanes[:2]:
#         candidates = lane.get("candidates", [])
#         if not candidates:
#             continue

#         first = candidates[0]
#         short = first.get("short_label")

#         suggested.append({
#             "label": f"🔀 Tôi đang xử lý {short}",
#             "utterance": f"tôi đang xử lý {short}"
#         })

#     suggested.append({
#         "label": "💬 Tôi mô tả thêm vấn đề",
#         "utterance": ""
#     })

#     # =========================
#     # Display policy
#     # UI đọc policy này để biết có show full runbook không
#     # =========================
#     display_policy = {
#         "show_primary_summary": True,
#         "show_full_runbook": bool(ctx.get("show_full_runbook", False)),
#         "show_suggested_utterances": True
#     }

#     return {
#         "message": message,
#         "suggested_utterances": suggested,
#         "display_policy": display_policy,
#         "turn_reason": turn_reason
#     }


def build_agent_result_candidate_from_context(
    query,
    ctx,
    turn_reason="candidate_presented"
):
    """
    ✅ CLEAN + SAFE VERSION
    - Giữ structure cho UI hiện tại
    - Bỏ lane logic
    - Chuẩn flow_state
    - Không phá hệ existing
    """

    # =============================
    # ✅ GET CANDIDATES (TIER1)
    # =============================
    candidates = ctx.get("current_lane_candidates", [])
    current_index = ctx.get("current_index", 0)

    if current_index >= len(candidates):
        current_index = 0
        ctx["current_index"] = 0

    primary = candidates[current_index] if candidates else None

    # =============================
    # ✅ SAVE CONTEXT
    # =============================
    ctx["current_primary"] = primary

    # =============================
    # ✅ MAP turn_reason → flow_state
    # =============================
    FLOW_MAP = {
        "candidate_presented": "fast_match",
        "open_runbook": "open_runbook",
        "next_primary": "next_primary"
    }

    flow_state = FLOW_MAP.get(turn_reason, "fast_match")
    ctx["flow_state"] = flow_state

    # =============================
    # ✅ BUILD ASSISTANT TURN
    # =============================
    assistant_turn = build_assistant_turn(
        ctx=ctx,
        flow_state=flow_state,
        user_input=query
    )

    # =============================
    # ✅ RETURN RESULT (GIỮ FORM CHO UI)
    # =============================
    return {
        "status": "candidate",

        "assistant_turn": assistant_turn,

        "data": {
            "primary": primary,

            # ✅ giữ lại để Phase 8 dùng
            "tier2_lanes": ctx.get("tier2_lanes", [])
        },

        "text": (
            primary.get("short_label")
            if primary else "Không tìm thấy runbook phù hợp"
        ),

        "meta": {
            "runbook_confirmed": False,
            "cache_policy": "no_write"
        },

        "trace": {
            "query": query,
            "flow_state": flow_state,
            "turn_reason": turn_reason,
            "current_index": current_index
        }
    }

def build_agent_result_candidate(query, tier1, tier2_lanes):

    primary = tier1[0] if tier1 else None

    return {
        "status": "candidate",

        "data": {
            "tier1": tier1,
            "tier2_lanes": tier2_lanes,
            "primary": primary
        },

        # UI hiển thị câu chính (Tier1 primary)
        "text": (
            primary["short_label"]
            if primary else "Không tìm thấy hướng xử lý phù hợp"
        ),

        "meta": {
            "runbook_confirmed": False,
            "cache_policy": "no_write"
        },

        "trace": {
            "query": query,
            "flow": "candidate"
        }
    }
##### CLEANUP CODE######################################
import unicodedata
import re

def normalize_vi(text: str) -> str:
    if not text:
        return ""

    # ✅ lower
    text = text.lower()

    # ✅ bỏ dấu tiếng Việt
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # ✅ bỏ ký tự đặc biệt
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # ✅ chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text

def resolve_intent_v2(user_input, ctx):

    text = normalize(user_input)

    score = {
        "open_runbook": 0,
        "next_primary": 0
    }

    # =============================
    # ✅ OPEN RUNBOOK SIGNALS
    # =============================
    if "runbook" in text:
        score["open_runbook"] += 3

    if "mo" in text or "xem" in text:
        score["open_runbook"] += 1

    # =============================
    # ✅ NEXT PRIMARY SIGNALS
    # =============================
    if "khac" in text or "khác" in text:
        score["next_primary"] += 2

    if "khong phai" in text or "sai" in text:
        score["next_primary"] += 3

    if "thu" in text or "doi" in text:
        score["next_primary"] += 1

    # =============================
    # ✅ CONTEXT BOOST (QUAN TRỌNG)
    # =============================
    if ctx.get("tier1"):

        # Nếu đang xem runbook mà user nói “khác”
        if "khac" in text:
            score["next_primary"] += 1

    # =============================
    # ✅ DECISION
    # =============================
    open_score = score["open_runbook"]
    next_score = score["next_primary"]

    print("🔍 INTENT SCORE:", score)

    if open_score >= next_score and open_score > 0:
        return "open_runbook"

    if next_score > open_score and next_score > 0:
        return "next_primary"

    return "stay"

def should_use_llm(flow_state: str) -> bool:
    """
    ✅ LLM chỉ dùng cho ambiguous / clarify / advisory
    ❌ KHÔNG dùng cho fast path
    """
    return flow_state in {
        "initial_ambiguous",
        "clarify_service",
        "clarify_issue_type",
        "clarify_error_message",
        "user_refinement",
        "related_direction",
        "advisory_tier2",
        "shift_detected",
        "shift_confirm"
    }


def build_template_message(ctx: dict, flow_state: str) -> str:
    """
    ✅ Message deterministic cho fast path
    ❌ KHÔNG gọi LLM
    """
    primary = ctx.get("current_primary") or {}
    title = primary.get("short_label", "runbook")

    if flow_state == "fast_match":
        return f"Đề xuất hiện tại: {title}."

    if flow_state == "open_runbook":
        return f"Đã mở runbook: {title}."

    if flow_state == "next_primary":
        return f"Tôi chuyển sang phương án khác trong cùng hướng: {title}."

    return "Đang xử lý ngữ cảnh hiện tại."


def build_suggested_utterances(ctx: dict, flow_state: str):
    """
    ✅ Action do agent quyết định (KHÔNG phải LLM)
    """
    actions = []

    # -------- FAST PATH --------
    if flow_state in {"fast_match", "next_primary"}:
        actions = [
            {
                "label": "📘 Mở runbook",
                "utterance": "mở runbook này",
                "action": "open_runbook"
            },
            {
                "label": "➡️ Đổi hướng dẫn khác",
                "utterance": "xem cái khác",
                "action": "next_primary"
            },
            # ✅ chuẩn bị cho Phase 8
            {
                "label": "💬 Tư vấn thêm",
                "utterance": "tư vấn thêm",
                "action": "advisory"
            }
        ]

    elif flow_state == "open_runbook":
        actions = [
            {
                "label": "➡️ Đổi phương án khác",
                "utterance": "xem cái khác",
                "action": "next_primary"
            }
        ]

    # -------- ADVISORY / CLARIFY --------
    elif flow_state in {
        "initial_ambiguous",
        "clarify_service",
        "clarify_issue_type",
        "clarify_error_message",
        "user_refinement",
        "advisory_tier2"
    }:
        # giữ minimal, không ép user
        actions = []

    return actions


def build_display_policy(ctx: dict, flow_state: str):
    """
    ✅ UI render policy
    """
    return {
        "show_primary_summary": True,
        "show_full_runbook": ctx.get("show_full_runbook", False)
    }


def build_assistant_turn(
    ctx: dict,
    flow_state: str,
    user_input: str = None,
):
    """
    ✅ CENTRAL FUNCTION
    ✅ KHÔNG được gọi LLM sai chỗ
    """

    actions = build_suggested_utterances(ctx, flow_state)

    # =============================
    # ✅ LLM CONTROL
    # =============================
    if should_use_llm(flow_state):
        # ⚠️ CHƯA IMPLEMENT LLM → tạm placeholder
        # → Phase 8 sẽ cắm vào đây

        message = "[LLM_MESSAGE_PLACEHOLDER]"

        # Ví dụ:
        # message = call_llm(
        #     build_llm_prompt_by_state(
        #         flow_state=flow_state,
        #         ctx=ctx,
        #         user_input=user_input
        #     )
        # )

    else:
        # ✅ FAST PATH → TEMPLATE
        message = build_template_message(ctx, flow_state)

    return {
        "flow_state": flow_state,              # ✅ VERY IMPORTANT for Phase 8
        "message": message,
        "suggested_utterances": actions,
        "display_policy": build_display_policy(ctx, flow_state)
    }

# ## hàm này xác định query có cần clarify không- chạy trước khi resolve issue
# def should_clarify(query: str):

#     text = normalize_vi(query)
#     tokens = text.split()

#     # =============================
#     # ✅ RULE 1: quá ngắn
#     # =============================
#     if len(tokens) <= 2:
#         return True

#     # =============================
#     # ✅ RULE 2: không có keyword IT
#     # =============================
#     keywords = [
#         "mail", "mailbox", "exchange",
#         "tai khoan", "user", "domain",
#         "queue", "database"
#     ]

#     if not any(k in text for k in keywords):
#         return True

#     # =============================
#     # ✅ CLEAR → không cần clarify
#     # =============================
#     return False


def detect_clarify_service(service_info):
    """
    ✅ PHÂ_service thật    ✅ PHÂN BIỆT:
    - multi_service thật
    - fake multi (noise)
    """

    resolved_service = service_info.get("resolved_service")
    candidates = service_info.get("candidate_services", [])
    confidence = service_info.get("confidence", 0.0)
    is_strong = service_info.get("is_strong", False)

    # =============================
    # ✅ CASE 1: NO SERVICE (QUAN TRỌNG NHẤT)
    # =============================
    # 👉 kể cả có candidate nhưng confidence quá thấp → coi là no_service
    if not resolved_service and confidence < 0.4:
        return "no_service"

    # =============================
    # ✅ CASE 2: MULTI SERVICE THẬT
    # =============================
    # 👉 phải có nhiều candidate + confidence đủ
    if len(candidates) > 1 and confidence >= 0.4 and not is_strong:
        return "multi_service"

    # =============================
    # ✅ CASE 3: RESOLVED / STRONG
    # =============================
    if resolved_service and is_strong:
        return None

    # =============================
    # ✅ DEFAULT → treat as no_service
    # =============================
    return "no_service"


def build_service_lock_result(query, ctx, service_info, clarify_reason):
    """
    Trả về response yêu cầu user chốt service.
    Không gọi LLM.
    """

    candidates = service_info.get("candidate_services", [])

    ctx["flow_state"] = FLOW_SERVICE_LOCK_PROMPT
    ctx["service_locked"] = False
    ctx["resolved_service"] = None
    ctx["service_candidates"] = candidates
    ctx["clarify_reason"] = clarify_reason
    ctx["original_query"] = query

    if clarify_reason == "no_service":
        message = (
            "Tôi chưa xác định rõ yêu cầu này thuộc dịch vụ nào. "
            "Bạn hãy chọn hoặc nhập tên dịch vụ đang xử lý để tôi tìm runbook đúng phạm vi."
        )
    else:
        message = (
            "Tôi thấy yêu cầu này có thể liên quan tới nhiều dịch vụ. "
            "Bạn hãy chốt giúp tôi dịch vụ chính để tôi tìm runbook đúng hướng."
        )

    actions = []

    for svc in candidates[:4]:
        actions.append({
            "label": f"🔎 {svc}",
            "utterance": svc,
            "action": "lock_service"
        })

    # fallback nếu resolver không có candidates
    if not actions:
        actions = [
            {
                "label": "Active Directory",
                "utterance": "Active Directory",
                "action": "lock_service"
            },
            {
                "label": "Exchange",
                "utterance": "Exchange",
                "action": "lock_service"
            },
            {
                "label": "VPN",
                "utterance": "VPN",
                "action": "lock_service"
            }
        ]

    return {
        "status": "candidate",

        "assistant_turn": {
            "flow_state": FLOW_SERVICE_LOCK_PROMPT,
            "message": message,
            "suggested_utterances": actions,
            "display_policy": {
                "show_primary_summary": False,
                "show_full_runbook": False
            }
        },

        "data": {
            "primary": None,
            "tier2_lanes": [],
            "service_candidates": candidates
        },

        "text": message,

        "meta": {
            "runbook_confirmed": False,
            "cache_policy": "no_write"
        },

        "trace": {
            "query": query,
            "flow_state": FLOW_SERVICE_LOCK_PROMPT,
            "clarify_reason": clarify_reason,
            "service_candidates": candidates
        }
    }

def try_lock_service_from_user(user_input, ctx):
    """
    User đang trả lời ở service_lock_prompt.
    Ưu tiên match với service_candidates.
    """

    text = normalize(user_input)
    candidates = ctx.get("service_candidates", [])

    for svc in candidates:
        if normalize(svc) in text or text in normalize(svc):
            return svc

    # Nếu user nhập trực tiếp service ngoài candidates, vẫn cho phép
    # vì mục tiêu Phase 8A là force service nhanh.
    if user_input and len(user_input.strip()) >= 2:
        return user_input.strip()

    return None

### START - các hàm sau xử lý khi lock service và user nhập thêm keyword
def normalize_issue_detection(raw_result):
    """
    Chuẩn hóa output từ issue_resolver.

    Output chuẩn:
    {
        "status": "none" | "partial" | "resolved",
        "issue_type": str | None,
        "issue_candidates": list[dict],
        "confidence": float
    }
    """

    if not raw_result or not isinstance(raw_result, dict):
        return {
            "status": "none",
            "issue_type": None,
            "issue_candidates": [],
            "confidence": 0.0
        }

    issue_type = (
        raw_result.get("issue_type")
        or raw_result.get("resolved_issue")
        or raw_result.get("intent")
        or raw_result.get("resolved_intent")
    )

    confidence = raw_result.get("confidence", raw_result.get("score", 0.0))

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    raw_candidates = (
        raw_result.get("issue_candidates")
        or raw_result.get("candidates")
        or raw_result.get("intents")
        or []
    )

    issue_candidates = []

    for i, c in enumerate(raw_candidates):
        if isinstance(c, str):
            issue_candidates.append({
                "issue_id": f"issue_{i}",
                "issue_label": c,
                "issue_hint": c,
                "confidence": 0.0
            })

        elif isinstance(c, dict):
            label = (
                c.get("issue_label")
                or c.get("label")
                or c.get("issue_type")
                or c.get("intent")
                or c.get("name")
            )

            if label:
                issue_candidates.append({
                    "issue_id": c.get("issue_id") or f"issue_{i}",
                    "issue_label": str(label),
                    "issue_hint": str(c.get("issue_hint") or c.get("hint") or label),
                    "confidence": float(c.get("confidence", c.get("score", 0.0)) or 0.0)
                })

    # =============================
    # Decision
    # =============================
    if issue_type and confidence >= 0.7:
        return {
            "status": "resolved",
            "issue_type": str(issue_type),
            "issue_candidates": issue_candidates,
            "confidence": confidence
        }

    if issue_candidates:
        return {
            "status": "partial",
            "issue_type": str(issue_type) if issue_type else None,
            "issue_candidates": issue_candidates,
            "confidence": confidence
        }

    return {
        "status": "none",
        "issue_type": None,
        "issue_candidates": [],
        "confidence": confidence
    }

def resolve_issue_for_locked_service(user_input, ctx, state):
    """
    Gọi issue_resolver trong scope service đã lock.
    Service đã được resolver ở Phase 8A.
    """

    service = ctx.get("resolved_service")
    raw_issue = resolve_issue_type(
        query=user_input,
        service=service,
        state=state
    )

    return normalize_issue_detection(raw_issue)

    # tách tier2_lanes sau khi có candidates từ issue resolver
def issue_candidates_to_tier2_lanes(issue_candidates):
    """
    Convert issue_candidates từ issue_resolver sang tier2_lanes stateful.
    Đây là tier2 của ISSUE layer, không phải tier2 của service retrieval.
    """

    lanes = []

    for i, issue in enumerate(issue_candidates or []):
        label = issue.get("issue_label") or issue.get("issue_type") or f"Hướng vấn đề {i + 1}"
        hint = issue.get("issue_hint") or label

        lane_id = issue.get("issue_id") or f"issue_lane_{i}"

        lanes.append({
            "lane_id": lane_id,
            "lane_label": str(label),
            "lane_hint": str(hint),
            "status": "new",
            "score": float(issue.get("confidence", 0.0) or 0.0),
            "candidates": [],
            "interaction": {
                "times_suggested": 0,
                "times_selected": 0,
                "times_failed": 0
            }
        })

    return normalize_tier2_lanes(lanes)

def build_issue_refine_retry(query, ctx):

    attempts = ctx.get("issue_refine_attempts", 0)
    service = ctx.get("resolved_service")

    # 🔥 TURN 1
    if attempts == 1:
        message = (
            f"Tôi đã xác định dịch vụ là: {service}.\n\n"
            "Tuy nhiên tôi vẫn chưa nhận ra rõ loại vấn đề bạn đang cần tìm.\n"
            "Bạn hãy mô tả thêm một vài từ khóa cụ thể hơn (ví dụ lỗi, thao tác, chức năng...)."
        )

    # 🔥 TURN 2 (deep hơn)
    elif attempts == 2:
        message = (
            f"Hiện tại tôi vẫn chưa xác định được chính xác vấn đề trong {service}.\n\n"
            "Bạn có thể cho biết cụ thể hơn:\n"
            "- Bạn đang thực hiện thao tác nào?\n"
            "- Có lỗi gì hiển thị không?\n"
            "- Liên quan mailbox, account hay gửi nhận mail?"
        )

    # 🔥 TURN >=3 → chuẩn bị reset
    else:
        message = (
            f"Tôi chưa có đủ thông tin để xác định vấn đề trong {service}.\n\n"
            "Bạn vui lòng mô tả lại rõ hơn hoặc thử nhập lại yêu cầu từ đầu."
        )

    return {
        "assistant_turn": {
            "flow_state": FLOW_ADVISORY_PRE_CANDIDATE,
            "message": message
        }
    }

def build_issue_refine_reset(ctx):
    """
    Reset nhẹ khi user không cung cấp đủ issue sau nhiều turn.
    """

    message = (
        "Tôi vẫn chưa thể xác định rõ nhóm vấn đề từ thông tin hiện tại. "
        "Bạn vui lòng mô tả lại yêu cầu từ đầu, gồm tên dịch vụ và lỗi/chức năng đang cần xử lý."
    )

    return {
        "status": "candidate",
        "assistant_turn": {
            "flow_state": "reset_required",
            "message": message,
            "suggested_utterances": [],
            "display_policy": {
                "show_primary_summary": False,
                "show_full_runbook": False
            }
        },
        "data": {
            "primary": None,
            "tier2_lanes": []
        },
        "text": message,
        "meta": {
            "runbook_confirmed": False,
            "cache_policy": "no_write"
        },
        "trace": {
            "flow_state": "reset_required",
            "reason": "max_issue_refine_attempts"
        }
    }

    

### END - Các hàm xử lý sau khi lock service và user nhập thêm keyword #####

def build_pre_candidate_advisory(query, ctx):
    """
    Phase 8B.1 — sau khi lock service nhưng chưa đủ intent
    Không retrieve
    Không tier2
    Không hard-code action
    """

    service = ctx.get("resolved_service", "dịch vụ")

    ctx["flow_state"] = FLOW_ADVISORY_PRE_CANDIDATE

    message = (
        f"Quá tốt! Vậy là chúng ta đã biết cần tìm Runbook cho dịch vụ: {service}. "
        "Giờ bạn hãy cho tôi biết thêm một vài thông tin hoặc từ khóa chính cho vấn đề bạn đang cần tìm, "
        "tôi sẽ hỗ trợ bạn chính xác hơn."
    )

    return {
        "status": "candidate",
        "assistant_turn": {
            "flow_state": FLOW_ADVISORY_PRE_CANDIDATE,
            "message": message,
            "suggested_utterances": [],  # ✅ không hard-code
            "display_policy": {
                "show_primary_summary": False,
                "show_full_runbook": False
            }
        },
        "data": {
            "primary": None
        },
        "text": message,
        "meta": {
            "runbook_confirmed": False,
            "cache_policy": "no_write"
        },
        "trace": {
            "query": query,
            "flow_state": FLOW_ADVISORY_PRE_CANDIDATE,
            "service": service
        }
    }

def is_enough_for_retrieval(query: str):
    tokens = query.split()
    return len(tokens) >= 5


def _safe_lane_id(value, index):
    """
    Tạo lane_id ổn định, không phụ thuộc LLM.
    """
    if not value:
        return f"lane_{index}"

    text = str(value).strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")

    # giữ đơn giản để tránh lỗi unicode
    if not text:
        return f"lane_{index}"

    return text[:80]

###### Phase 8B.2 - Nếu refine không match thì vào LLM  + Tier2 #########
def normalize_tier2_lanes(raw_lanes):
    """
    Chuẩn hóa tier2_lanes về schema stateful.
    Dùng cho toàn hệ từ Phase 8B.2 trở đi.
    """

    normalized = []

    for i, lane in enumerate(raw_lanes or []):

        # =============================
        # Case 1: lane là string
        # =============================
        if isinstance(lane, str):
            label = lane.strip()

            normalized.append({
                "lane_id": _safe_lane_id(label, i),
                "lane_label": label,
                "lane_hint": label,
                "status": "new",
                "score": 0.0,
                "candidates": [],
                "interaction": {
                    "times_suggested": 0,
                    "times_selected": 0,
                    "times_failed": 0
                }
            })
            continue

        # =============================
        # Case 2: lane là dict
        # =============================
        if not isinstance(lane, dict):
            continue

        lane_label = (
            lane.get("lane_label")
            or lane.get("label")
            or lane.get("name")
            or lane.get("lane_id")
            or f"Hướng liên quan {i + 1}"
        )

        lane_id = lane.get("lane_id") or _safe_lane_id(lane_label, i)

        lane_hint = (
            lane.get("lane_hint")
            or lane.get("hint")
            or lane.get("description")
            or lane_label
        )

        candidates = lane.get("candidates") or []

        interaction = lane.get("interaction") or {}

        normalized.append({
            "lane_id": lane_id,
            "lane_label": str(lane_label),
            "lane_hint": str(lane_hint),
            "status": lane.get("status", "new"),
            "score": float(lane.get("score", 0.0) or 0.0),
            "candidates": candidates,
            "interaction": {
                "times_suggested": int(interaction.get("times_suggested", 0) or 0),
                "times_selected": int(interaction.get("times_selected", 0) or 0),
                "times_failed": int(interaction.get("times_failed", 0) or 0),
            }
        })

    return normalized

def merge_tier2_lanes(existing_lanes, new_lanes):
    """
    Merge tier2_lanes mới vào tier2_lanes cũ.
    Giữ lại status / interaction cũ nếu lane_id trùng.
    """

    existing = normalize_tier2_lanes(existing_lanes)
    new = normalize_tier2_lanes(new_lanes)

    by_id = {}

    for lane in existing:
        by_id[lane["lane_id"]] = lane

    for lane in new:
        lane_id = lane["lane_id"]

        if lane_id in by_id:
            old = by_id[lane_id]

            # cập nhật thông tin mới nhưng giữ state conversation
            old["lane_label"] = lane.get("lane_label", old["lane_label"])
            old["lane_hint"] = lane.get("lane_hint", old["lane_hint"])
            old["score"] = lane.get("score", old.get("score", 0.0))
            old["candidates"] = lane.get("candidates", old.get("candidates", []))
        else:
            by_id[lane_id] = lane

    return list(by_id.values())

def get_lanes_for_advisory(ctx, limit=5):
    """
    Chọn các lane nên hiển thị trong advisory.
    Ưu tiên lane mới, tránh lane rejected.
    Đồng thời update status/times_suggested.
    """

    lanes = normalize_tier2_lanes(ctx.get("tier2_lanes", []))

    # Ưu tiên new trước, rồi suggested/explored nếu không đủ
    candidates = [
        lane for lane in lanes
        if lane.get("status") != "rejected"
    ]

    candidates.sort(
        key=lambda x: (
            0 if x.get("status") == "new" else 1,
            -float(x.get("score", 0.0) or 0.0)
        )
    )

    selected = candidates[:limit]

    selected_ids = set()

    for lane in selected:
        selected_ids.add(lane["lane_id"])

        if lane.get("status") == "new":
            lane["status"] = "suggested"

        lane["interaction"]["times_suggested"] += 1

    # ghi lại vào ctx
    updated = []

    for lane in lanes:
        if lane["lane_id"] in selected_ids:
            # lấy bản đã update trong selected
            updated_lane = next(
                x for x in selected
                if x["lane_id"] == lane["lane_id"]
            )
            updated.append(updated_lane)
        else:
            updated.append(lane)

    ctx["tier2_lanes"] = updated

    return selected

def mark_lane_mentioned_by_user(ctx, user_input):
    """
    Nếu user nhập text gần với lane_label/lane_hint thì đánh dấu lane explored.
    Không dùng để quyết định runbook.
    """

    if not user_input:
        return None

    text = normalize(user_input)

    lanes = normalize_tier2_lanes(ctx.get("tier2_lanes", []))

    matched_lane = None

    for lane in lanes:
        label = normalize(lane.get("lane_label", ""))
        hint = normalize(lane.get("lane_hint", ""))

        if label and label in text:
            matched_lane = lane
            break

        # match nhẹ theo từng từ quan trọng trong label
        label_tokens = [t for t in label.split() if len(t) >= 3]
        if label_tokens and any(t in text for t in label_tokens):
            matched_lane = lane
            break

        if hint and hint in text:
            matched_lane = lane
            break

    if matched_lane:
        matched_lane["status"] = "explored"
        matched_lane["interaction"]["times_selected"] += 1

        # ghi lại ctx
        for i, lane in enumerate(lanes):
            if lane["lane_id"] == matched_lane["lane_id"]:
                lanes[i] = matched_lane
                break

        ctx["tier2_lanes"] = lanes

    return matched_lane

def build_advisory_tier2(query, ctx):

    service = ctx.get("resolved_service") or "dịch vụ hiện tại"

    lanes = ctx.get("tier2_lanes", [])[:4]

    ctx["flow_state"] = FLOW_ADVISORY_TIER2
    ctx["show_full_runbook"] = False

    # =========================
    # ✅ MESSAGE
    # =========================
    if lanes:

        lines = []
        for lane in lanes:
            label = lane.get("lane_label")
            hint = lane.get("lane_hint")

            if hint and hint != label:
                lines.append(f"- {label}: {hint}")
            else:
                lines.append(f"- {label}")

        lane_text = "\n".join(lines)

        message = (
            f"Có vẻ thông tin hiện tại vẫn chưa đủ để xác định chính xác runbook trong {service}.\n\n"
            f"Tôi thấy có thể bạn đang gặp một trong các hướng sau:\n{lane_text}\n\n"
            "Bạn có thể chọn một hướng bên dưới hoặc mô tả rõ thêm để tôi tìm chính xác hơn."
        )

    else:
        message = (
            f"Tôi chưa xác định rõ vấn đề trong {service}.\n\n"
            "Bạn có thể mô tả thêm một vài từ khóa cụ thể hơn về lỗi hoặc thao tác đang thực hiện."
        )

    # =========================
    # ✅ ACTIONS (QUAN TRỌNG)
    # =========================
    actions = []

    for lane in lanes:
        actions.append({
            "label": lane.get("lane_label"),
            "utterance": lane.get("query_expansion"),  # 🔥 dùng cho refine
            "action": "select_lane"
        })

    return {

        "status": "candidate",

        "assistant_turn": {
            "flow_state": FLOW_ADVISORY_TIER2,
            "message": message,
            "suggested_utterances": actions,   # ✅ FIX
            "display_policy": {
                "show_primary_summary": False,
                "show_full_runbook": False
            }
        },

        "data": {
            "primary": None,
            "tier2_lanes": ctx.get("tier2_lanes", [])
        },

        "text": message,

        "meta": {
            "runbook_confirmed": False,
            "cache_policy": "no_write"
        },

        "trace": {
            "query": query,
            "flow_state": FLOW_ADVISORY_TIER2,
            "service": service,
            "tier2_count": len(ctx.get("tier2_lanes", []))
        }
    }


# =====================================================
# MAIN
# =====================================================
import time
def run_agent(session_id, user_input, vector_store):
 
    # =============================
    # INIT STATE
    # =============================
    state = get_session(session_id)
    ensure_state(state)

    ctx = state["candidate_context"]
    print("🔥 RUN:", user_input, "|", time.time())
    print("🧪 [ENTRY]")
    print("CTX FULL:", ctx)
    print("FLOW_STATE:", ctx.get("flow_state"))
    print("ADVISORY_ASKED:", ctx.get("advisory_asked"))
    print("ORIGINAL:", ctx.get("original_query"))
    print("USER_INPUT:", user_input)
    print("------")

    # =============================
    # NORMALIZE INPUT
    # =============================
    normalized = normalize_vi(user_input)

    print("RAW:", user_input)
    print("NORMALIZED:", normalized)

    # =====================================================
    # ✅ PHASE 8A — SERVICE LOCK (UNIFIED)
    # =====================================================

    # 👉 nếu CHƯA lock service
    if not ctx.get("service_locked"):

        print("🔍 [8A] Detecting service...")

        service_raw = resolve_service_v2(
            user_input,
            state=state,
            use_llm=False
        )

        service_info = normalize_service_detection(service_raw)
        clarify_reason = detect_clarify_service(service_info)

        # ✅ cần clarify → hỏi user chọn service
        if clarify_reason:

            ctx["flow_state"] = FLOW_SERVICE_LOCK_PROMPT

            return build_service_lock_result(
                query=user_input,
                ctx=ctx,
                service_info=service_info,
                clarify_reason=clarify_reason
            )

        # ✅ service rõ → lock luôn
        resolved_service = service_info.get("resolved_service")

        ctx["resolved_service"] = resolved_service
        ctx["service_locked"] = True

        print("✅ [8A] SERVICE LOCKED:", resolved_service)

        # 👉 chuyển sang B1, KHÔNG retrieval
        ctx["flow_state"] = FLOW_ADVISORY_PRE_CANDIDATE
        ctx["query_history"] = []
        ctx["refine_attempts"] = 0

        return {
            "assistant_turn": {
                "flow_state": FLOW_ADVISORY_PRE_CANDIDATE,
                "message": (
                    f"Đã xác định dịch vụ: {resolved_service}.\n\n"
                    "Bạn hãy nhập từ khóa để tìm runbook."
                )
            }
        }


    # =====================================================
    # ✅ 8A — SERVICE LOCK PROMPT (CLARIFY STEP)
    # =====================================================
    if ctx.get("flow_state") == FLOW_SERVICE_LOCK_PROMPT:

        print("🔍 [8A] Waiting user chọn service...")

        locked_service = try_lock_service_from_user(user_input, ctx)

        if locked_service:

            print("✅ [8A] SERVICE LOCKED FROM USER:", locked_service)

            ctx["resolved_service"] = locked_service
            ctx["service_locked"] = True

            ctx["flow_state"] = FLOW_ADVISORY_PRE_CANDIDATE
            ctx["query_history"] = []
            ctx["refine_attempts"] = 0

            return {
                "assistant_turn": {
                    "flow_state": FLOW_ADVISORY_PRE_CANDIDATE,
                    "message": (
                        f"Đã xác định dịch vụ: {locked_service}.\n\n"
                        "Bạn hãy nhập từ khóa để tìm runbook."
                    )
                }
            }

        # ❗ user chưa chọn đúng → hỏi lại
        return build_service_lock_result(
            query=user_input,
            ctx=ctx,
            service_info={
                "candidate_services": ctx.get("service_candidates", []),
                "resolved_service": None,
                "confidence": 0.0,
                "is_strong": False
            },
            clarify_reason=ctx.get("clarify_reason", "no_service")
        )
    
    # =========================
    # ✅ G5 — CONFIRM SERVICE SWITCH
    # =========================
    if user_input == "yes_switch_service":
        new_service = ctx.get("pending_service_switch")

        print("✅ [G5] CONFIRM SWITCH:", new_service)

        ctx["resolved_service"] = new_service
        ctx["service_locked"] = True

        ctx["query_history"] = []
        ctx["tier1"] = []
        ctx["tier2_lanes"] = []

        ctx["pending_service_switch"] = None
        ctx["flow_state"] = FLOW_ADVISORY_PRE_CANDIDATE

        return {
            "assistant_turn": {
                "message": f"Đã chuyển sang dịch vụ {new_service}. Bạn nhập lại từ khóa."
            }
        }

    if user_input == "no_switch_service":

        print("↩️ [G5] REJECT SWITCH")

        ctx["pending_service_switch"] = None
        ctx["flow_state"] = FLOW_ADVISORY_TIER2

        return {
            "assistant_turn": {
                "message": "OK, tôi sẽ tiếp tục với dịch vụ hiện tại."
            }
        }
    
    
    
    
    # =====================================================
    # ✅ INPUT ROUTER — B2 CONTEXT
    # =====================================================
    if ctx.get("flow_state") == FLOW_ADVISORY_TIER2:

        print("🧠 [B2 ROUTER] INPUT:", user_input)

        # =========================
        # ✅ G3 — SELECT LANE / REFINE
        # =========================
        print("🧭 [G3] REFINE")

        selected_keyword = user_input.strip()

        ctx["pending_refine_keyword"] = selected_keyword
        ctx["flow_state"] = FLOW_ADVISORY_PRE_CANDIDATE

        return run_agent(
            session_id,
            selected_keyword,
            vector_store
        )


    # =====================================================
    # PHASE 8B.1 — ISSUE-FIRST RESOLVER
    # =====================================================
    if ctx.get("flow_state") == FLOW_ADVISORY_PRE_CANDIDATE:

        print("🧪 [INSIDE 8B.1]")

        service = ctx.get("resolved_service")

        if not service:
            return None

        # ====================================
        # 1. ACCUMULATE KEYWORD
        # ====================================
        
        if "query_history" not in ctx or not isinstance(ctx["query_history"], list):
            ctx["query_history"] = []

        new_keyword = user_input.strip()
        
        pending = ctx.pop("pending_refine_keyword", None)

        if pending:
            new_keyword = pending

        ctx["query_history"].append(new_keyword)
        ctx["query_history"] = ctx["query_history"][-3:]

        combined_query = " ".join(ctx["query_history"])

        print("🧪 COMBINED QUERY:", combined_query)

        # ====================================
        # 2. MULTI QUERY RETRIEVAL
        # ====================================
        candidate_result = retrieve_candidates_multiquery(
            combined_query,
            state,
            vector_store,
            RUNBOOK_DATA
        )

        tier1 = candidate_result.get("tier1_candidates", [])
        tier2_lanes = normalize_tier2_lanes(
            candidate_result.get("tier2_lanes", [])
        )
        strong_tier1 = candidate_result.get("strong_tier1", [])


        # ====================================
        # 🔥 G4 — DRIFT DETECT (CORRECT DESIGN)
        # ====================================

        current_service = ctx.get("resolved_service")

        top_candidate = None

        if strong_tier1:
            top_candidate = strong_tier1[0]
        elif tier1:
            top_candidate = tier1[0]

        if top_candidate:
            candidate_service = top_candidate.get("service_id")

            print("🧪 [G4] RETRIEVAL SERVICE:", candidate_service, "| current:", current_service)

            if candidate_service and candidate_service != current_service:

                print("🔄 [G4] DRIFT DETECTED VIA RETRIEVAL")

                ctx["pending_service_switch"] = candidate_service
                ctx["flow_state"] = FLOW_CONFIRM_SERVICE_SWITCH

                return {
                    "assistant_turn": {
                        "flow_state": FLOW_CONFIRM_SERVICE_SWITCH,
                        "message": (
                            f"Kết quả phù hợp nhất lại thuộc dịch vụ {candidate_service}.\n\n"
                            f"Hiện tại bạn đang ở {current_service}.\n\n"
                            "Bạn có muốn chuyển sang dịch vụ này để tiếp tục không?"
                        ),
                        "suggested_utterances": [
                            {
                                "label": f"✅ Chuyển sang {candidate_service}",
                                "utterance": "yes_switch_service"
                            },
                            {
                                "label": "❌ Giữ nguyên dịch vụ hiện tại",
                                "utterance": "no_switch_service"
                            }
                        ]
                    }
                }

        # ====================================
        # 3. DECISION
        # ====================================

        # ✅ CASE 1 — STRONG MATCH → FAST
        if strong_tier1:
            print("✅ [8B.1] STRONG MATCH → FAST")

            ctx["tier1"] = strong_tier1
            ctx["tier2_lanes"] = tier2_lanes
            ctx["current_lane_candidates"] = strong_tier1   # ✅ fix
            ctx["current_index"] = 0
            ctx["flow_state"] = FLOW_FAST_MATCH

            return build_agent_result_candidate_from_context(
                query=combined_query,
                ctx=ctx,
                turn_reason="candidate_presented"
            )


        # ✅ CASE 2 — WEAK MATCH (tier1 có nhưng không strong)
        # ✅ CASE 2 — WEAK MATCH → B2
        if tier1:
            print("⚠️ [8B.1] WEAK MATCH → B2")

            ctx["tier1"] = tier1
            ctx["tier2_lanes"] = tier2_lanes
            ctx["flow_state"] = FLOW_ADVISORY_TIER2

            return build_advisory_tier2(
                query=combined_query,
                ctx=ctx
            )


        # ✅ CASE 3 — NO MATCH
        print("❌ [8B.1] NO MATCH")

        ctx["refine_attempts"] = ctx.get("refine_attempts", 0) + 1

        if ctx["refine_attempts"] >= MAX_ISSUE_REFINE_ATTEMPTS:

            print("❌ MAX REFINE → RESET")

            ctx["query_history"] = []
            ctx["refine_attempts"] = 0
            ctx["flow_state"] = None
            ctx["service_locked"] = False
            ctx["resolved_service"] = None

            return {
                "assistant_turn": {
                    "message": "Tôi chưa xác định được vấn đề. Bạn vui lòng mô tả lại từ đầu."
                }
            }

        print("⚠️ NO MATCH → ASK MORE")

        return {
            "assistant_turn": {
                "flow_state": FLOW_ADVISORY_PRE_CANDIDATE,
                "message": (
                    "Tôi chưa tìm thấy runbook phù hợp.\n\n"
                    "Bạn có thể nhập thêm từ khóa chi tiết hơn?"
                )
            }
        }

    # =========================================
    # Phase 8B.2 — ISSUE ADVISORY LOOP
    # =========================================
    




    # =====================================================
    # ✅ PHASE 1 — ASSIST MODE (ĐÃ CÓ CONTEXT)
    # =====================================================
    if ctx.get("tier1"):

        print("🧠 ASSIST MODE ACTIVE")

        tier1 = ctx.get("tier1", [])
        current_index = ctx.get("current_index", 0)

        intent = resolve_intent_v2(user_input, ctx)
        print("✅ INTENT:", intent)

        # -------------------------
        # ✅ OPEN RUNBOOK
        # -------------------------
        if intent == "open_runbook":

            ctx["show_full_runbook"] = True

            return build_agent_result_candidate_from_context(
                query=ctx.get("original_query"),
                ctx=ctx,
                turn_reason="open_runbook"
            )

        # -------------------------
        # ✅ NEXT PRIMARY
        # -------------------------
        if intent == "next_primary":

            if tier1:
                current_index = (current_index + 1) % len(tier1)

                ctx["current_index"] = current_index

            ctx["show_full_runbook"] = False

            return build_agent_result_candidate_from_context(
                query=ctx.get("original_query"),
                ctx=ctx,
                turn_reason="next_primary"
            )

        # -------------------------
        # ✅ DEFAULT (STAY)
        # -------------------------
        return build_agent_result_candidate_from_context(
            query=ctx.get("original_query"),
            ctx=ctx,
            turn_reason="candidate_presented"
        )

    