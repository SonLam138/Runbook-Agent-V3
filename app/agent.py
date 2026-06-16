import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.service_resolver import resolve_service
from app.llm import call_llm
from app.session_store import get_session, update_session, reset_session
from app.tool import retrieve_candidates_meta, search_RB_topk

from app.load_model import embedding_model as CACHE_MODEL
from app.load_knowledge import get_clarify_knowledge, get_decision_knowledge

# =====================================================
# CONFIG
# =====================================================
MAX_CLARIFY_TURNS = 3
MAX_RETRY_RUNBOOKS = 3
CACHE_THRESHOLD = 0.82

# Nếu top score thấp hơn ngưỡng này → KHÔNG được trả runbook
MIN_CANDIDATE_CONFIDENCE = 0.6

# Nếu top score cao hơn ngưỡng này → search luôn, không cần LLM
STRONG_MATCH_THRESHOLD = 0.7

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

#CACHE_MODEL_PATH = Path(r"D:\bge-m3")

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


def detect_strong_match_by_score(full_candidates, threshold=STRONG_MATCH_THRESHOLD):
    """
    If top-1 FAISS score is strong enough -> search directly without LLM decision
    """
    if not full_candidates:
        return False, None

    top = full_candidates[0]
    score = top.get("score", 0)

    print(f"🔎 TOP SCORE: {score:.4f}")

    if score >= threshold:
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
    query = state.get("semantic_query")
    tried = state.get("tried_runbooks", [])

    if not query:
        return reply(
            session_id,
            state,
            "❌ Tôi chưa đủ thông tin để thử lại. Bạn mô tả rõ hơn giúp tôi."
        )

    results = search_RB_topk(
        query=query,
        exclude_titles=tried,
        topk=3
    )

    if not results:
        return reply(session_id, state, "❌ Không tìm thấy hướng khác")

    rb = results[0]

    remember_success(state, query, rb)
    state["last_result_status"] = "retry_returned"

    return reply(
        session_id,
        state,
        "⚠️ Thử phương án khác:\n\n" + format_runbook(rb)
    )


# =====================================================
# MAIN
# =====================================================
"""
def run_agent(session_id, user_input):
    state = get_session(session_id)
    ensure_state(state)

    # append user message
    state["history"].append({"role": "user", "text": user_input})

    # 0) semantic cache
    cached = check_semantic_cache(user_input)
    if cached:
        print("⚡ SEMANTIC CACHE HIT")
        return reply(
            session_id,
            state,
            "✅ (từ semantic cache)\n\n" + format_runbook(cached)
        )

    # 1) failure handling
    if is_failure(user_input, state):
        return handle_retry(session_id, state)

    # 2) clarifying block
    if state["mode"] == "clarifying":
        fill_pending_slot(state, user_input)
        state["clarify_turns"] += 1

        if state["clarify_turns"] > MAX_CLARIFY_TURNS:
            reset_session(session_id)
            return "❌ Tôi chưa đủ thông tin. Bạn vui lòng đặt lại câu hỏi rõ hơn."

        effective_query = build_semantic_query(state)
    else:
        effective_query = user_input

    # 3) save current semantic query
    if state["mode"] == "clarifying" or not state.get("semantic_query"):
        state["semantic_query"] = effective_query

    # 4) retrieve candidates
    full_candidates, meta_candidates = retrieve_candidates_meta(
        query=effective_query,
        state=state
    )

    # 5) confidence gate: score quá thấp -> ask_more
    # CHỈ áp dụng confidence gate khi KHÔNG đang clarify
    if state["mode"] == "idle" and not is_candidate_confident(full_candidates):
        print("⚠️ LOW CONFIDENCE (idle) → ASK MORE")

        start_clarify(state, user_input, "issue_type")

        msg = larify_message_llmgenerate_c(user_input, state, "issue_type")
        if not msg:
            msg = generate_clarify_message_fallback("issue_type", user_input)

        return reply(session_id, state, msg)

    # 6) strong match -> direct search
    match, idx = detect_strong_match_by_score(full_candidates)
    if match:
        rb = full_candidates[idx]

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # 7) LLM decision
    d = decide_with_candidates(user_input, state, meta_candidates)

    if d.get("action") == "search":
        idx = d.get("selected_index", 1)
        try:
            idx = int(idx) - 1
        except Exception:
            idx = 0

        if idx < 0 or idx >= len(full_candidates):
            idx = 0

        rb = full_candidates[idx]

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # 8) ask_more
    next_slot = d.get("next_slot") or "issue_type"

    # 🔥 CLARIFY: luôn ưu tiên LLM riêng, KHÔNG dùng message chung chung từ decision
    msg = generate_clarify_message_llm(user_input, state, next_slot)

    # fallback nếu LLM clarify fail
    if not msg or not msg.strip():
        msg = generate_clarify_message_fallback(next_slot, user_input)

    # safety net cuối cùng
    if not msg:
        msg = "Bạn có thể mô tả rõ hơn vấn đề bạn đang gặp không?"

    start_clarify(state, user_input, next_slot)

    return reply(session_id, state, msg)
"""

def run_agent(session_id, user_input):
    state = get_session(session_id)
    ensure_state(state)

    # append user message
    state["history"].append({"role": "user", "text": user_input})

    # 0) semantic cache
    cached = check_semantic_cache(user_input)
    if cached:
        print("⚡ SEMANTIC CACHE HIT")
        return reply(
            session_id,
            state,
            "✅ (từ semantic cache)\n\n" + format_runbook(cached)
        )

    # 0.5) resolve service early
    service_info = resolve_service(user_input, state)
    if service_info:
        state["resolved_service"] = service_info["service"]
        print(
            f"🧭 RESOLVED SERVICE: {service_info['service']} "
            f"(score={service_info['score']:.3f}, source={service_info['source']}, "
            f"strong={service_info['is_strong']})"
        )

        # nếu resolver đủ mạnh thì fill luôn service slot
        if service_info.get("is_strong") and not state["slots"].get("service"):
            state["slots"]["service"] = service_info["service"]

    # 1) failure handling
    if is_failure(user_input, state):
        return handle_retry(session_id, state)

    # 2) clarifying block
    if state["mode"] == "clarifying":
        fill_pending_slot(state, user_input)
        state["clarify_turns"] += 1

        if state["clarify_turns"] > MAX_CLARIFY_TURNS:
            reset_session(session_id)
            return "❌ Tôi chưa đủ thông tin. Bạn vui lòng đặt lại câu hỏi rõ hơn."

        # nếu resolver đã biết service mà slot service chưa có, tự fill luôn
        if not state["slots"].get("service") and state.get("resolved_service"):
            state["slots"]["service"] = state["resolved_service"]

        effective_query = build_semantic_query(state)
    else:
        effective_query = user_input

    # 3) save current semantic query
    if state["mode"] == "clarifying" or not state.get("semantic_query"):
        state["semantic_query"] = effective_query

    # 4) retrieve candidates
    full_candidates, meta_candidates = retrieve_candidates_meta(
        query=effective_query,
        state=state
    )

    # 5) confidence gate: score quá thấp -> ask_more
    # CHỈ áp dụng confidence gate khi KHÔNG đang clarify
    if state["mode"] == "idle" and not is_candidate_confident(full_candidates):
        print("⚠️ LOW CONFIDENCE (idle) → ASK MORE")

        start_clarify(state, user_input, "issue_type")

        # dùng LLM clarify riêng
        msg = generate_clarify_message_llm(user_input, state, "issue_type")
        if not msg:
            msg = generate_clarify_message_fallback("issue_type", user_input)

        return reply(session_id, state, msg)

    # 6) strong match -> direct search
    match, idx = detect_strong_match_by_score(full_candidates)
    if match:
        rb = full_candidates[idx]

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # 7) LLM decision
    d = decide_with_candidates(user_input, state, meta_candidates)

    if d.get("action") == "search":
        idx = d.get("selected_index", 1)
        try:
            idx = int(idx) - 1
        except Exception:
            idx = 0

        if idx < 0 or idx >= len(full_candidates):
            idx = 0

        rb = full_candidates[idx]

        remember_success(state, effective_query, rb)
        state["mode"] = "idle"
        state["pending_slot"] = None

        return reply(session_id, state, format_runbook(rb))

    # 8) ask_more
    next_slot = d.get("next_slot") or "issue_type"

    # Clarify: luôn ưu tiên LLM riêng, không dùng message generic từ decision
    msg = generate_clarify_message_llm(user_input, state, next_slot)

    if not msg or not msg.strip():
        msg = generate_clarify_message_fallback(next_slot, user_input)

    if not msg:
        msg = "Bạn có thể mô tả rõ hơn vấn đề bạn đang gặp không?"

    start_clarify(state, user_input, next_slot)

    return reply(session_id, state, msg)