import json
import re
import numpy as np
import faiss
from app.vector_store import FaissStore
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.service_resolver import resolve_service
from app.llm import call_llm
from app.session_store import get_session, update_session, reset_session
#from app.tool import retrieve_candidates_meta, search_RB_topk
from app.vector_store import ChromaStore
from app.load_model import embedding_model as CACHE_MODEL
from app.load_knowledge import get_clarify_knowledge, get_decision_knowledge
from app.vector_store import ChromaStore
from app.load_chroma import load_chroma
from app.issue_type_resolver import resolve_issue_type


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


with open("data/runbook_data.json", "r", encoding="utf-8") as f:
    RUNBOOK_DATA = json.load(f)

# build index lookup nhanh
RUNBOOK_INDEX = {
    rb["title"]: rb
    for rb in RUNBOOK_DATA
}


chroma_collection = load_chroma()
vector_store = ChromaStore(collection=chroma_collection)



print("✅ VECTOR STORE BACKEND:", type(vector_store).__name__)
print("✅ CHROMA COLLECTION LOADED:", chroma_collection is not None)

# =====================================================
# ENRICH RUNBOOK FROM JSON
# =====================================================
def enrich_runbook_from_json(rb):
    """
    Nhận rb từ Chroma (metadata),
    trả lại full runbook từ JSON
    """

    title = rb.get("title")

    full_rb = RUNBOOK_INDEX.get(title)

    if not full_rb:
        print(f"⚠️ Không tìm thấy runbook trong JSON: {title}")
        return rb  # fallback

    return full_rb



# =====================================================
# SEMANTIC CACHE (GLOBAL IN-MEMORY)
# =====================================================
#CACHE_MODEL = SentenceTransformer(str(CACHE_MODEL_PATH), trust_remote_code=True)

SEMANTIC_CACHE_DATA = []      # [{"query":..., "runbook":...}]
SEMANTIC_CACHE_VECTORS = []   # [vector, vector, ...]
SEMANTIC_CACHE_INDEX = None   # faiss index


# =====================================================
# SEMANTIC CACHE FUNCTIONS
# =====================================================
def rebuild_semantic_cache_index():
    global SEMANTIC_CACHE_INDEX

    if not SEMANTIC_CACHE_VECTORS:
        SEMANTIC_CACHE_INDEX = None
        return

    vectors = np.array(SEMANTIC_CACHE_VECTORS, dtype="float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    SEMANTIC_CACHE_INDEX = index


def save_feedback(query, runbook):
    """
    Save semantic cache from user feedback (e.g. UI like button).
    """
    global SEMANTIC_CACHE_DATA, SEMANTIC_CACHE_VECTORS

    if not query or not runbook:
        return

    vec = CACHE_MODEL.encode(
        [query],
        normalize_embeddings=True
    )[0]

    SEMANTIC_CACHE_DATA.append({
        "query": query,
        "runbook": runbook
    })

    SEMANTIC_CACHE_VECTORS.append(vec)
    rebuild_semantic_cache_index()


def check_semantic_cache(query, threshold=CACHE_THRESHOLD):
    """
    Semantic cache lookup by FAISS.
    """
    if SEMANTIC_CACHE_INDEX is None or not SEMANTIC_CACHE_DATA:
        return None

    q_vec = CACHE_MODEL.encode(
        [query],
        normalize_embeddings=True
    )
    q_vec = np.array(q_vec, dtype="float32")

    scores, indices = SEMANTIC_CACHE_INDEX.search(q_vec, 1)

    score = float(scores[0][0])
    idx = int(indices[0][0])

    print(f"🧠 SEMANTIC CACHE SCORE: {score:.4f}")

    if idx == -1:
        return None

    if score >= threshold:
        return SEMANTIC_CACHE_DATA[idx]["runbook"]

    return None


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
    Dùng LLM để sinh câu hỏi clarify thông minh theo slot còn thiếu.
    Đã inject Knowledge Grounding theo service đã resolve.
    """
    service = state.get("resolved_service")
    knowledge = get_clarify_knowledge(service)

    slot_guide = {
        "issue_type": "Hỏi rõ vấn đề hoặc loại lỗi user đang gặp.",
        "service": "Hỏi rõ user đang thao tác trên hệ thống/dịch vụ nào.",
        "error_message": "Hỏi rõ thông báo lỗi hoặc mã lỗi cụ thể."
    }

    known_issue = state["slots"].get("issue_type")
    known_service = state["slots"].get("service")
    known_error = state["slots"].get("error_message")

    prompt = f"""
{knowledge}

Bạn là IT support agent nội bộ.

User vừa nói:
"{user_input}"

Trạng thái hội thoại hiện tại:
- issue_type: {known_issue}
- service: {known_service}
- error_message: {known_error}
- resolved_service: {service}

Slot còn thiếu cần hỏi tiếp:
- next_slot: {next_slot}
- hướng dẫn: {slot_guide.get(next_slot, "Hỏi làm rõ thêm thông tin cần thiết.")}

YÊU CẦU:
1. Chỉ hỏi đúng 1 câu ngắn gọn bằng tiếng Việt.
2. Phải tuân theo knowledge ở trên.
3. Không hỏi chung chung kiểu:
   - "Bạn mô tả rõ hơn"
   - "Cho tôi biết thêm"
4. Nếu đã biết service hoặc context, hãy hỏi sâu hơn theo service đó.
5. Không giải thích, không liệt kê, chỉ trả đúng 1 câu hỏi.

QUAN TRỌNG:
- KHÔNG được hỏi lại thông tin user đã cung cấp
- Nếu user đã nói rõ loại lỗi, phải hỏi sâu hơn, không lặp lại
"""

    try:
        msg = call_llm(prompt)
        msg = (msg or "").strip()

        if msg:
            msg = msg.split("\n")[0].strip()

            # nếu câu vẫn generic thì coi như fail
            if is_generic_clarify_message(msg):
                print("⚠️ Clarify rejected as generic:", msg)
                return None

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
# SUCCESS MEMORY
# =====================================================
def remember_success(state, final_query, rb):
    if not rb:
        return

    title = rb.get("title")
    if title and title not in state["tried_runbooks"]:
        state["tried_runbooks"].append(title)

    state["last_runbook"] = rb
    state["last_result_status"] = "returned"
    state["last_action"] = "search"
    state["semantic_query"] = final_query

    state["semantic_cache"].append({
        "semantic_query": final_query,
        "runbook_title": rb.get("title"),
        "service": rb.get("service")
    })
    state["semantic_cache"] = state["semantic_cache"][-5:]


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
def handle_retry(session_id, state):
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

def run_agent(session_id, user_input):
    state = get_session(session_id)
    ensure_state(state)

    # append user message
    state["history"].append({"role": "user", "text": user_input})

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

            confirmed_service = None

            # Case 1: user xác nhận service hiện tại là đúng
            if "đúng" in user_text or "phải" in user_text or "ok" in user_text:
                if resolved:
                    confirmed_service = resolved

            # Case 2: user nói service khác → resolve lại service từ câu trả lời
            else:
                new_service_info = resolve_service(user_input, state)

                if new_service_info:
                    confirmed_service = new_service_info.get("service")

                    if confirmed_service:
                        print(f"🔁 SERVICE UPDATED → {confirmed_service}")

            # Nếu xác định được service thì update state
            if confirmed_service:
                state["slots"]["service"] = confirmed_service
                state["resolved_service"] = confirmed_service

                next_slot = "issue_type"

                #msg = generate_clarify_message_llm(user_input, state, next_slot)
                #if not msg:
                #    msg = generate_clarify_message_fallback(next_slot, user_input)
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

            return reply(session_id, state, msg)

        # =====================================================
        # ISSUE TYPE RESOLUTION
        # =====================================================
        if pending_slot == "issue_type":
            service = state["slots"].get("service") or state.get("resolved_service")

            # Nếu vì lý do nào đó chưa có service chắc chắn → quay lại confirm service
            if not service:
                msg = generate_clarify_message_fallback("service", user_input)
                start_clarify(state, user_input, "service")
                return reply(session_id, state, msg)

            result = resolve_issue_type(user_input, service, state)

            print("🧩 ISSUE TYPE RESULT:", result)

            # Nếu match keyword → lưu issue_type và SEARCH NGAY
            if result.get("issue_type"):
                state["slots"]["issue_type"] = result["issue_type"]

                query = f"{service} {result['issue_type']}"
                state["semantic_query"] = query

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

                    return reply(session_id, state, format_runbook(rb))

                # Có keyword nhưng search chưa đủ mạnh → hỏi error_message
                next_slot = "error_message"

                #msg = "generate_clarify_message_llm(user_input, state, next_slot)"
                msg = "Bạn có nhận được thông báo lỗi hoặc mã lỗi nào không? Nếu có, bạn hãy nhập lên đây để tôi có thể tìm kiếm chính xác hơn."
                #if not msg:
                #    msg = generate_clarify_message_fallback(next_slot, user_input)

                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Không match keyword → hỏi error_message để refine
            if result.get("needs_more"):
                next_slot = "error_message"

                msg = "Bạn có nhận được thông báo lỗi hoặc mã lỗi nào không? Nếu có, bạn hãy nhập lên đây để tôi có thể tìm kiếm chính xác hơn."
                #msg = generate_clarify_message_llm(user_input, state, next_slot)
                #if not msg:
                #    msg = generate_clarify_message_fallback(next_slot, user_input)

                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Safety fallback
            msg = generate_clarify_message_fallback("error_message", user_input)
            start_clarify(state, user_input, "error_message")
            return reply(session_id, state, msg)

        # =====================================================
        # ERROR MESSAGE
        # =====================================================
        if pending_slot == "error_message":
            service = state["slots"].get("service") or state.get("resolved_service")
            issue_type = state["slots"].get("issue_type")

            # Lấy thông tin lỗi user vừa nhập
            new_error_msg = user_input.strip()
            # 🔥 BACKFILL ISSUE_TYPE từ error_message nếu chưa có
            service = state["slots"].get("service") or state.get("resolved_service")

            # chỉ backfill nếu hiện tại chưa có issue_type
            if not state["slots"].get("issue_type") and service:
                retry_result = resolve_issue_type(new_error_msg, service, state)

                if retry_result and retry_result.get("issue_type"):
                    state["slots"]["issue_type"] = retry_result["issue_type"]
                    print("🔁 ISSUE TYPE BACKFILLED:", retry_result["issue_type"])

            # Nếu trước đó đã có error_message thì cộng dồn thêm,
            # vì user có thể bổ sung thông tin qua nhiều lượt clarify.
            old_error_msg = state["slots"].get("error_message")

            if old_error_msg:
                combined_error_msg = f"{old_error_msg} {new_error_msg}".strip()
            else:
                combined_error_msg = new_error_msg

            state["slots"]["error_message"] = combined_error_msg

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

            # Search trước, chỉ trả runbook nếu strong match
            match, idx = detect_strong_match_by_score(full_candidates)

            if match:
                rb = enrich_runbook_from_json(full_candidates[idx])

                remember_success(state, query, rb)
                state["mode"] = "idle"
                state["pending_slot"] = None

                return reply(session_id, state, format_runbook(rb))

            # Nếu chưa match mà vẫn còn lượt clarify thì hỏi thêm thông tin lỗi
            if state["clarify_turns"] < MAX_CLARIFY_TURNS:
                next_slot = "error_message"

                msg = generate_clarify_message_llm(user_input, state, next_slot)

                if not msg:
                    msg = (
                        "Bạn có thể cung cấp thêm thông báo lỗi, mã lỗi "
                        "hoặc thao tác cụ thể trước khi lỗi xảy ra không?"
                    )

                start_clarify(state, user_input, next_slot)

                return reply(session_id, state, msg)

            # Nếu đã hết lượt clarify mà vẫn không match → dừng và reset
            reset_session(session_id)
            return "❌ Tôi chưa đủ thông tin để xác định runbook phù hợp."

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
    if is_failure(user_input, state):
        return handle_retry(session_id, state)

    # 2.2) semantic cache
    cached = check_semantic_cache(user_input)
    if cached:
        print("⚡ SEMANTIC CACHE HIT")
        remember_success(state, user_input, cached)
        state["last_result_status"] = "cache_returned"
        state["last_action"] = "search"
        return reply(
            session_id,
            state,
            "✅ (từ semantic cache)\n\n" + format_runbook(cached)
        )
    
    # 2.3) resolve service early
    service_info = resolve_service(user_input, state)
    if service_info:
        state["resolved_service"] = service_info["service"]
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
        # if service_info.get("is_strong") and not state["slots"].get("service"):
        #     state["slots"]["service"] = service_info["service"]



    # 2.4) first search
    effective_query = user_input
    state["semantic_query"] = effective_query

    full_candidates, meta_candidates = vector_store.retrieve_candidates(
        query=effective_query,
        state=state
    )
    # 2.5) STRONG MATCH → RETURN NGAY
    match, idx = detect_strong_match_by_score(full_candidates)

    if match:
        rb = enrich_runbook_from_json(full_candidates[idx])

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # =====================================================
    # 3) FIRST SEARCH DECISION
    # =====================================================
    # Query đầu tiên LUÔN được search.
    # Nhưng chỉ trả runbook nếu strong match.
    match, idx = detect_strong_match_by_score(full_candidates)

    if match:
        rb = enrich_runbook_from_json(full_candidates[idx])

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # =====================================================
    # 4) NOT STRONG → SERVICE CONFIRM
    # =====================================================
    print("⚠️ FIRST SEARCH NOT STRONG → SERVICE CONFIRM")

    start_clarify(state, user_input, "service")

    resolved = state.get("resolved_service")

    if resolved:
        msg = f"Tôi hiểu bạn đang gặp vấn đề trên hệ thống {resolved}. Đúng không?"
    else:
        msg = generate_clarify_message_fallback("service", user_input)

    return reply(session_id, state, msg)


