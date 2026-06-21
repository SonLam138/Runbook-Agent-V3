import json
from pathlib import Path
import re
import numpy as np
import faiss
#from app.vector_store import FaissStore
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.service_resolver import resolve_service
from app.llm import call_llm
from app.session_store import get_session, update_session, reset_session
#from app.tool import retrieve_candidates_meta, search_RB_topk
from app.vector_store import ChromaStore
from app.load_model import embedding_model as CACHE_MODEL
from app.load_knowledge import get_clarify_knowledge, get_decision_knowledge
#from app.vector_store import ChromaStore
#from app.load_chroma import load_chroma
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
# NORMALIZE SERVICE RESOLVE RESULT
# ============================================================
def normalize_service_result(result):
    """
    Chuẩn hóa output từ resolve_service/extract_service về format thống nhất.

    Return:
        {
            "service": str | None,
            "confidence": float | None,
            "is_strong": bool
        }
    """

    service = None
    confidence = None

    if result is None:
        return {
            "service": None,
            "confidence": None,
            "is_strong": False,
        }

    # Case 1: resolve_service trả string
    if isinstance(result, str):
        service = result.strip()
        confidence = 1.0 if service else None

    # Case 2: resolve_service trả tuple/list: (service, confidence)
    elif isinstance(result, (tuple, list)):
        if len(result) >= 1:
            service = result[0]
        if len(result) >= 2:
            confidence = result[1]

    # Case 3: resolve_service trả dict
    elif isinstance(result, dict):
        service = (
            result.get("service")
            or result.get("resolved_service")
            or result.get("value")
        )
        confidence = (
            result.get("confidence")
            or result.get("score")
            or result.get("service_confidence")
        )

    if isinstance(service, str):
        service = service.strip()

    if not service:
        return {
            "service": None,
            "confidence": confidence,
            "is_strong": False,
        }

    # Nếu chưa có confidence nhưng đã match deterministic thì coi là strong
    if confidence is None:
        confidence = 1.0

    try:
        confidence_float = float(confidence)
    except Exception:
        confidence_float = None

    is_strong = confidence_float is None or confidence_float >= 0.75

    return {
        "service": service,
        "confidence": confidence_float,
        "is_strong": is_strong,
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

# =====================================================
# MAIN
# =====================================================

def run_agent(session_id, user_input, vector_store):
    state = get_session(session_id)
    ensure_state(state)

    # append user message
    state["history"].append({"role": "user", "text": user_input})
    # LOGGING
    log_event(
        event="agent_start",
        session_id=session_id,
        data={
            "user_input": user_input
        },
        decision_trace={
            "action": "start",
            "reason": "new_query_received",
            "confidence": 1.0
        }
    )


    # =====================================================
    # 1) CLARIFYING FLOW
    # =====================================================
    # Nếu đang trong chế độ clarify thì xử lý riêng và return luôn.
    # KHÔNG cho rơi xuống idle search flow bên dưới.
    if state["mode"] == "clarifying":

        print("🟡 CLARIFY MODE ACTIVE:", state.get("pending_slot"))

        pending_slot = state.get("pending_slot")
        state["clarify_turns"] += 1

        # quá số lượt hỏi → stop
        if state["clarify_turns"] > MAX_CLARIFY_TURNS:
            reset_session(session_id)
            return "❌ Tôi chưa đủ thông tin để xác định runbook phù hợp. Bạn vui lòng đặt lại câu hỏi rõ hơn."

        # =====================================================
        # SERVICE CONFIRM
        # =====================================================
        if pending_slot == "service":
            resolved = state.get("resolved_service")
            user_text = normalize(user_input)
            log_service_clarify_input(
                session_id=session_id,
                state=state,
                user_input=user_input
            )
            confirmed_service = None

            # Case 1: user xác nhận service hiện tại là đúng
            if "đúng" in user_text or "phải" in user_text or "ok" in user_text:
                if resolved:
                    confirmed_service = resolved
                    log_service_clarify_extract(
                        session_id=session_id,
                        state=state,
                        user_input=user_input,
                        extracted_service=resolved,
                        confidence=1.0,
                        result="success",
                        reason="user_confirm_existing_service"
                    )

            # Case 2: user nói service khác → resolve lại service từ câu trả lời
            else:
                new_service_info = resolve_service(user_input, state)
                log_service_clarify_extract(
                    session_id=session_id,
                    state=state,
                    user_input=user_input,
                    extracted_service=new_service_info.get("service") if new_service_info else None,
                    confidence=new_service_info.get("score") if new_service_info else None,
                    result="success" if new_service_info and new_service_info.get("service") else "fail",
                    reason="user_provided_new_service"
                )

                if new_service_info:
                    confirmed_service = new_service_info.get("service")

                    if confirmed_service:
                        print(f"🔁 SERVICE UPDATED → {confirmed_service}")

            # Nếu xác định được service thì update state
            if confirmed_service:
                old_service = state["slots"].get("service")

                state["slots"]["service"] = confirmed_service
                log_service_clarify_update(
                    session_id=session_id,
                    state=state,
                    old_service=old_service,
                    new_service=confirmed_service,
                    next_slot="issue_type",
                    reason="service_confirmed_or_updated"
                )
                state["resolved_service"] = confirmed_service

                next_slot = "issue_type"

                service = state.get("resolved_service") or state["slots"].get("service")

                msg = f"""Chúng tôi cần thêm chút thông tin để tìm đúng runbook cho bạn. Bạn vui lòng mô tả yêu cầu/vấn đề có chứa các từ khóa liên quan.

                Ví dụ:
                - Lỗi mailbox --> 1 số từ khóa : không đăng nhập được/lỗi gửi mail/dung lượng mailbox...
                - Lỗi đăng nhập --> 1 số từ khóa : đăng nhập máy tính/đăng nhập ứng dụng A,B/MFA...
                Từ khóa càng gần yêu cầu/vấn đề của bạn thì kết quả tìm kiếm sẽ càng chính xác."""

                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Nếu vẫn chưa xác nhận được service → hỏi lại service
            msg = generate_clarify_message_fallback("service", user_input)

            start_clarify(state, user_input, "service")

            log_service_clarify_retry(
                session_id=session_id,
                state=state,
                user_input=user_input,
                reason="cannot_confirm_service"
            )

            return reply(session_id, state, msg)

        # =====================================================
        # ISSUE TYPE RESOLUTION
        # =====================================================
        if pending_slot == "issue_type":
            service = state["slots"].get("service") or state.get("resolved_service")
            log_event(
                event="clarify_input",
                session_id=session_id,
                data={
                    "flow_context": "clarify",
                    "slot": "issue_type",
                    "user_input": user_input,
                    "collected_slots": state.get("slots", {}).copy(),
                },
                decision_trace={
                    "action": "receive_input",
                    "reason": "user_response_issue_type",
                    "confidence": 1.0
                }
            )

            # Nếu vì lý do nào đó chưa có service chắc chắn → quay lại confirm service
            if not service:
                msg = generate_clarify_message_fallback("service", user_input)
                start_clarify(state, user_input, "service")
                return reply(session_id, state, msg)

            result = resolve_issue_type(user_input, service, state)
            log_issue_type_resolve_attempt(
                session_id=session_id,
                state=state,
                user_input=user_input,
                service=service,
                result=result
            )

            print("🧩 ISSUE TYPE RESULT:", result)

            # Nếu match keyword → lưu issue_type và SEARCH NGAY
            if result.get("issue_type"):
                state["slots"]["issue_type"] = result["issue_type"]
                log_issue_type_selected(
                    session_id=session_id,
                    state=state,
                    service=service,
                    issue_type=result["issue_type"]
                )

                query = f"{service} {result['issue_type']}"
                state["semantic_query"] = query
                log_clarify_query_built(
                    session_id=session_id,
                    state=state,
                    query=query,
                    service=service,
                    issue_type=result["issue_type"]
                )

                full_candidates, meta_candidates = vector_store.retrieve_candidates(
                    query=query,
                    state=state
                )

                match, idx = detect_strong_match_by_score(full_candidates)

                if match:
                    rb = enrich_runbook_from_json(full_candidates[idx])

                    remember_success(state, query, rb)
                    state["mode"] = "idle"
                    state["pending_slot"] = None
                    log_clarify_strong_match(
                        session_id=session_id,
                        state=state,
                        query=query,
                        selected_rb=full_candidates[idx]
                    )

                    return reply(session_id, state, format_runbook(rb))

                # Có keyword nhưng search chưa đủ mạnh → hỏi error_message
                next_slot = "error_message"

                msg = "Bạn có nhận được thông báo lỗi hoặc mã lỗi nào không? Nếu có, bạn hãy nhập lên đây để tôi có thể tìm kiếm chính xác hơn."
                log_clarify_next_issue_type(
                    session_id=session_id,
                    state=state,
                    from_slot=state.get("pending_slot"),
                    to_slot=next_slot,
                    reason="match_not_strong_after_issue_type"
                )
                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Không match keyword → hỏi error_message để refine
            if result.get("needs_more"):
                next_slot = "error_message"
                log_clarify_next_issue_type(
                    session_id=session_id,
                    state=state,
                    from_slot=state.get("pending_slot"),   # ✅ FIX
                    to_slot=next_slot,                     # ✅ FIX
                    reason="issue_type_not_clear"
                )

                msg = "Bạn có nhận được thông báo lỗi hoặc mã lỗi nào không? Nếu có, bạn hãy nhập lên đây để tôi có thể tìm kiếm chính xác hơn."
                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Safety fallback
            msg = generate_clarify_message_fallback("error_message", user_input)
            log_clarify_next_issue_type(
                session_id=session_id,
                state=state,
                reason="fallback"
            )
            start_clarify(state, user_input, "error_message")
            return reply(session_id, state, msg)

        # =====================================================
        # ERROR MESSAGE
        # =====================================================
        if pending_slot == "error_message":
            service = state["slots"].get("service") or state.get("resolved_service")
            issue_type = state["slots"].get("issue_type")
            # Dừng lại nếu bị troll
            user_text = normalize(user_input)
            log_event(
                event="clarify_input",
                session_id=session_id,
                data={
                    "flow_context": "clarify",
                    "slot": "error_message",
                    "user_input": user_input,
                    "collected_slots": state.get("slots", {}).copy(),
                },
                decision_trace={
                    "action": "receive_input",
                    "reason": "user_response_error_message",
                    "confidence": 1.0
                }
            )

            if "không" in user_text and (
                "mã lỗi" in user_text
                or "ma loi" in user_text
                or "error" in user_text
                or "thông báo lỗi" in user_text
                or "thong bao loi" in user_text
            ):
                # Graceful exit khỏi clarify flow, KHÔNG reset session.
                state["mode"] = "idle"
                state["pending_slot"] = None
                state["last_result_status"] = "clarify_stopped_no_error_message"
                state["last_action"] = "clarify_stop"

                return reply(
                    session_id,
                    state,
                    "❌ Không có mã lỗi hoặc thông tin bổ sung, tôi chưa thể xác định runbook phù hợp.\n"
                    "👉 Bạn vui lòng mô tả rõ hơn theo hướng: thao tác nào bị lỗi, lỗi xảy ra ở màn hình/bước nào, "
                    "hoặc chọn một nhóm vấn đề gần nhất như: không đăng nhập được, lỗi gửi/nhận, tài khoản bị khóa, dung lượng, phân quyền..."
                )

            # Lấy thông tin lỗi user vừa nhập
            new_error_msg = user_input.strip()
            old_error_msg = state["slots"].get("error_message")

            if old_error_msg:
                combined_error_msg = f"{old_error_msg} {new_error_msg}".strip()
            else:
                combined_error_msg = new_error_msg

            state["slots"]["error_message"] = combined_error_msg
            # 🔥 BACKFILL ISSUE_TYPE từ error_message nếu chưa có
            service = state["slots"].get("service") or state.get("resolved_service")

            # chỉ backfill nếu hiện tại chưa có issue_type
            if not state["slots"].get("issue_type") and service:
                retry_result = resolve_issue_type(new_error_msg, service, state)

                if retry_result and retry_result.get("issue_type"):
                    state["slots"]["issue_type"] = retry_result["issue_type"]
                    print("🔁 ISSUE TYPE BACKFILLED:", retry_result["issue_type"])
            issue_type = state["slots"].get("issue_type")  # ✅ refresh lại

            # Nếu trước đó đã có error_message thì cộng dồn thêm,
            # vì user có thể bổ sung thông tin qua nhiều lượt clarify.


            # Build query từ đủ 3 slot:
            # service + issue_type + error_message
            query_parts = []

            if service:
                query_parts.append(service)

            if issue_type:
                query_parts.append(issue_type)

            if combined_error_msg:
                query_parts.append(combined_error_msg)

            query = " ".join(query_parts)
            state["semantic_query"] = query

            print("🔎 ERROR MESSAGE SEARCH QUERY:", query)

            full_candidates, meta_candidates = vector_store.retrieve_candidates(
                query=query,
                state=state
            )
            log_event(
                event="retrieval_done",
                session_id=session_id,
                data={
                    "flow_context": "clarify_search",
                    "query": query,
                    "candidate_count": len(full_candidates) if full_candidates else 0,
                    "top_k": [
                        {
                            "rank": i + 1,
                            "title": rb.get("title"),
                            "score": rb.get("score", 0.0)
                        }
                        for i, rb in enumerate(full_candidates[:3])
                    ] if full_candidates else []
                },
                decision_trace={
                    "action": "retrieve",
                    "reason": "clarify_search",
                    "confidence": full_candidates[0].get("score", 0.0) if full_candidates else 0.0
                }
            )

            # Search trước, chỉ trả runbook nếu strong match
            match, idx = detect_strong_match_by_score(full_candidates)

            if match:
                rb = enrich_runbook_from_json(full_candidates[idx])

                remember_success(state, query, rb)
                state["mode"] = "idle"
                state["pending_slot"] = None

                log_clarify_strong_match(
                    session_id=session_id,
                    state=state,
                    query=query,
                    selected_rb=full_candidates[idx]
                )


                return reply(session_id, state, format_runbook(rb))

            # Nếu chưa match mà vẫn còn lượt clarify thì hỏi thêm thông tin lỗi
            if state["clarify_turns"] < MAX_CLARIFY_TURNS:
                next_slot = "error_message"

                msg = generate_clarify_message_llm(user_input, state, next_slot)

                if not msg:
                    service = state["slots"].get("service") or state.get("resolved_service")

                    if service:
                        msg = (
                            f"Với {service}, lỗi xảy ra ở thao tác hoặc chức năng nào, "
                            "ví dụ đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
                        )
                    else:
                        msg = (
                            "Lỗi xảy ra ở thao tác hoặc chức năng CNTT nào, "
                            "ví dụ đăng nhập, phân quyền, kết nối, gửi/nhận, dung lượng hoặc đồng bộ?"
                        )
                log_clarify_next_issue_type(
                    session_id=session_id,
                    state=state,
                    from_slot=state.get("pending_slot"),
                    to_slot=next_slot,
                    reason="match_not_strong_after_issue_type"
                )

                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Nếu đã hết lượt clarify mà vẫn không match → graceful exit, KHÔNG reset session.
            state["mode"] = "idle"
            state["pending_slot"] = None
            state["last_result_status"] = "clarify_max_turns_reached"
            state["last_action"] = "clarify_stop"
            
            log_event(
                event="clarify_stop",
                session_id=session_id,
                data={
                    "flow_context": "clarify",
                    "reason": "max_turns_reached",
                    "collected_slots": state.get("slots", {}).copy()
                },
                decision_trace={
                    "action": "clarify_stop",
                    "reason": "max_turns_reached",
                    "confidence": 0.0
                }
            )
            return reply(
                session_id,
                state,
                "❌ Tôi chưa đủ thông tin để xác định runbook phù hợp. "
                "Bạn có thể đặt lại câu hỏi với dịch vụ + nhóm lỗi cụ thể hơn, ví dụ: Exchange lỗi gửi mail, Active Directory reset password, VPN không đăng nhập được."
            )

        # =====================================================
        # UNKNOWN CLARIFY STATE
        # =====================================================
        reset_session(session_id)
        return "❌ Trạng thái làm rõ chưa hợp lệ. Bạn vui lòng đặt lại câu hỏi."

    # =====================================================
    # 2) IDLE FLOW
    # =====================================================
    # Từ đây trở xuống chỉ chạy khi state["mode"] != "clarifying"

    # 2.1) failure handling
    if is_failure(user_input, state, ):
        return handle_retry(session_id, state, vector_store)

    # 2.2) semantic cache
    cached = check_semantic_cache(user_input)

    if cached:
        print("⚡ SEMANTIC CACHE HIT")

        log_event(
            event="cache_hit",
            session_id=session_id,
            data={
                "query_type": "initial",
                "query": user_input,
                "runbook_id": cached.get("runbook_id") if isinstance(cached, dict) else None,
                "runbook_title": cached.get("title") if isinstance(cached, dict) else None
            },
            decision_trace={
                "action": "cache_hit",
                "reason": "semantic_cache_matched_before_vector_search",
                "confidence": 1.0
            }
        )

        remember_success(state, user_input, cached)
        state["last_result_status"] = "cache_returned"
        state["last_action"] = "search"

        return reply(
            session_id,
            state,
            "✅ (từ semantic cache)\n\n" + format_runbook(cached)
        )

    #✅ MISS → log ngay sau if
    log_event(
        event="cache_miss",
        session_id=session_id,
        data={
            "query_type": "initial",
            "query": user_input
        },
        decision_trace={
            "action": "cache_miss",
            "reason": "no_semantic_cache_match_continue_to_vector_search",
            "confidence": 0.0
        }
    )
    
    # 2.3) resolve service early
    service_info = resolve_service(user_input, state)
    
    # ✅ ADD LOG: service_resolve_attempt
    log_service_resolve_attempt(
        session_id=session_id,
        state=state,
        user_input=user_input,
        resolved_service=service_info.get("service") if service_info else None,
        confidence=service_info.get("score") if service_info else None,
        result="success" if service_info and service_info.get("is_strong") else "fail",
        reason="idle_flow_service_resolution"
    )

    if service_info:
        state["resolved_service"] = service_info["service"]
        
        if service_info.get("is_strong"):
            state["slots"]["service"] = service_info["service"]

        print(
            f"🧭 RESOLVED SERVICE: {service_info['service']} "
            f"(score={service_info['score']:.3f}, source={service_info['source']}, "
            f"strong={service_info['is_strong']})"
        )

        # QUAN TRỌNG:
        # Không fill state["slots"]["service"] ở đây nữa.
        # resolved_service chỉ là guess.
        # slots["service"] chỉ được fill sau khi user confirm.
        #

    # 2.4) first search
    effective_query = user_input
    state["semantic_query"] = effective_query
    # Logging
    log_event(
    event="query_prepared",
    session_id=session_id,
    data={
        "query_type": "initial",
        "raw_query": user_input,
        "semantic_query": effective_query
    },
    decision_trace={
        "action": "prepare_query",
        "reason": "initial_search",
        "confidence": 1.0
    }
    )
    full_candidates, meta_candidates = vector_store.retrieve_candidates(
        query=effective_query,
        state=state
    )

    log_event(
    event="retrieval_done",
    session_id=session_id,
    data={
        "query_type": "initial",
        "query": effective_query,
        "candidate_count": len(full_candidates) if full_candidates else 0,
        "top_k": [
            {
                "rank": i + 1,
                "runbook_id": rb.get("runbook_id"),
                "title": rb.get("title"),
                "score": rb.get("score", 0.0)
            }
            for i, rb in enumerate(full_candidates[:3])
        ] if full_candidates else []
    },
    decision_trace={
        "action": "retrieve",
        "reason": "initial_vector_search_completed",
        "confidence": (
            full_candidates[0].get("score", 0.0)
            if full_candidates else 0.0
        )
    }
    )
    # 2.5) STRONG MATCH → RETURN NGAY
    match, idx = detect_strong_match_by_score(full_candidates)

    if match:
        rb = enrich_runbook_from_json(full_candidates[idx])
        selected = full_candidates[idx]
        log_event(
            event="strong_match_found",
            session_id=session_id,
            data={
                "flow_context": "initial_search",
                "query_type": "initial",

                # TRAINING SIGNAL: raw input
                "user_input": user_input,

                # TRAINING SIGNAL: query dùng để search
                "semantic_query": effective_query,

                # TRAINING SIGNAL: selected result
                "selected_title": selected.get("title"),
                "selected_service": selected.get("service"),
                "selected_score": selected.get("score", 0.0),
                "candidate_index": idx,

                # TRAINING SIGNAL: top-k candidates (rất quan trọng cho retrieval tuning)
                "top_k": [
                    {
                        "rank": i + 1,
                        "title": rb.get("title"),
                        "service": rb.get("service"),
                        "score": rb.get("score", 0.0)
                    }
                    for i, rb in enumerate(full_candidates[:3])
                ],

                # TRAINING SIGNAL: label (positive case)
                "label": "positive_match"
            },
            decision_trace={
                "action": "return_runbook",
                "reason": "strong_match trong lan query dau tien",
                "confidence": selected.get("score", 0.0)
            }
        )

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # # =====================================================
    # # 3) FIRST SEARCH DECISION
    # # =====================================================
    # # Query đầu tiên LUÔN được search.
    # # Nhưng chỉ trả runbook nếu strong match.
    # match, idx = detect_strong_match_by_score(full_candidates)

    # if match:
    #     rb = enrich_runbook_from_json(full_candidates[idx])

    #     remember_success(state, effective_query, rb)
    #     state["mode"] = "idle"
    #     state["pending_slot"] = None

    #     return reply(session_id, state, format_runbook(rb))

    # =====================================================
    # 4) NOT STRONG → SERVICE CONFIRM
    # =====================================================
    print("⚠️ FIRST SEARCH NOT STRONG → SERVICE CONFIRM")
    best = full_candidates[0] if full_candidates else None
    log_event(
    event="low_confidence",
    session_id=session_id,
    data={
        "flow_context": "initial_search",
        "query_type": "initial",

        # TRAINING SIGNAL
        "user_input": user_input,
        "semantic_query": effective_query,

        # TRAINING SIGNAL: best candidate nhưng chưa đủ mạnh
        "best_title": best.get("title") if isinstance(best, dict) else None,
        "best_service": best.get("service") if isinstance(best, dict) else None,
        "best_score": best.get("score", 0.0) if isinstance(best, dict) else 0.0,

        # TRAINING SIGNAL: agent quyết định hỏi gì
        "next_action": "clarify_service",
        "missing_slot": "service",

        # TRAINING SIGNAL: label
        "label": "needs_clarification"
    },
    decision_trace={
        "action": "clarify",
        "reason": "initial_search_not_strong",
        "confidence": best.get("score", 0.0) if isinstance(best, dict) else 0.0
    }
)

    if not service_info or not service_info["is_strong"]:
 
        start_clarify(state, user_input, "service")
        log_service_clarify_start(
            session_id=session_id,
            state=state,
            reason="initial_search_not_strong_and_no_strong_service"
        ) 
        return reply(
            session_id,
            state,
            "Bạn đang gặp vấn đề với dịch vụ nào?"
        )

    else:
        # ✅ service đã rõ -> KHÔNG hỏi lại

        state["mode"] = "clarifying"
        state["pending_slot"] = "issue_type"

        return reply(
            session_id,
            state,
            f"Bạn đang gặp vấn đề gì trên {service_info['service']}?"
        )


