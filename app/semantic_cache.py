from app.load_model import embedding_model as CACHE_MODEL
import json
import re
import numpy as np
import faiss

# =====================================================
# SEMANTIC CACHE (GLOBAL IN-MEMORY)
# =====================================================
#CACHE_MODEL = SentenceTransformer(str(CACHE_MODEL_PATH), trust_remote_code=True)

SEMANTIC_CACHE_DATA = []      # [{"query":..., "runbook":...}]
SEMANTIC_CACHE_VECTORS = []   # [vector, vector, ...]
SEMANTIC_CACHE_INDEX = None   # faiss index
SEMANTIC_CACHE_ENTRIES = []
CACHE_THRESHOLD = 0.82


RUNBOOK_DATA = None

def load_runbook_data():
    global RUNBOOK_DATA

    if RUNBOOK_DATA is None:
        with open("data/runbook_data.json", "r", encoding="utf-8") as f:
            RUNBOOK_DATA = json.load(f)

    return RUNBOOK_DATA


RUNBOOK_INDEX = None

def get_runbook_index():
    global RUNBOOK_INDEX

    if RUNBOOK_INDEX is None:
        data = load_runbook_data()

        RUNBOOK_INDEX = {
            rb["title"]: rb
            for rb in data
        }

    return RUNBOOK_INDEX

# =====================================================
# ENRICH RUNBOOK FROM JSON
# =====================================================
def enrich_runbook_from_json(rb):
    """
    Nhận rb từ Chroma (metadata),
    trả lại full runbook từ JSON
    """

    if not rb:
        return rb

    title = rb.get("title")

    index = get_runbook_index()

    full_rb = index.get(title)

    if not full_rb:
        print(f"⚠️ Không tìm thấy runbook trong JSON: {title}")
        return rb  # fallback

    return full_rb

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

    # =========================
    # 1) guard condition
    # =========================
    if SEMANTIC_CACHE_INDEX is None or not SEMANTIC_CACHE_ENTRIES:
        return None

    # =========================
    # 2) encode query (giữ nguyên của anh)
    # =========================
    q_vec = CACHE_MODEL.encode(
        [query],
        normalize_embeddings=True
    )
    q_vec = np.array(q_vec, dtype="float32")

    # =========================
    # 3) search FAISS
    # =========================
    scores, indices = SEMANTIC_CACHE_INDEX.search(q_vec, 1)

    score = float(scores[0][0])
    idx = int(indices[0][0])

    print(f"🧠 SEMANTIC CACHE SCORE: {score:.4f}")

    # =========================
    # 4) validate index
    # =========================
    if idx < 0 or idx >= len(SEMANTIC_CACHE_ENTRIES):
        return None

    # =========================
    # 5) threshold check
    # =========================
    if score >= threshold:
        entry = SEMANTIC_CACHE_ENTRIES[idx]

        if isinstance(entry, dict):
            return entry.get("runbook")

    return None
# =========================
# Add successful query → runtime FAISS cache
# =========================
def add_to_runtime_semantic_cache(semantic_query, rb):

    global SEMANTIC_CACHE_VECTORS
    global SEMANTIC_CACHE_ENTRIES

    if not semantic_query or not rb:
        return

    title = rb.get("title")
    service = rb.get("service")

    # =========================
    # 1) avoid duplicate in runtime cache
    # =========================
    for item in SEMANTIC_CACHE_ENTRIES:
        if (
            item.get("semantic_query") == semantic_query
            and item.get("runbook_title") == title
        ):
            return

    # =========================
    # 2) embed query ⚠️ dùng đúng function hiện tại của anh
    # =========================
    vector = CACHE_MODEL.encode(semantic_query)

    # =========================
    # 3) append dữ liệu
    # =========================
    SEMANTIC_CACHE_VECTORS.append(vector)

    SEMANTIC_CACHE_ENTRIES.append({
        "semantic_query": semantic_query,
        "runbook_title": title,
        "service": service,
        "runbook": rb
    })

    # =========================
    # 4) rebuild index
    # =========================
    rebuild_semantic_cache_index()

# =====================================================
# SUCCESS MEMORY
# =====================================================
def remember_success(state, final_query, rb):
    if not rb:
        return

    title = rb.get("title")
    service = rb.get("service")

    # =========================
    # 1) keep old behavior
    # =========================
    if title and title not in state["tried_runbooks"]:
        state["tried_runbooks"].append(title)

    state["last_runbook"] = rb
    state["last_result_status"] = "returned"
    state["last_action"] = "search"
    state["semantic_query"] = final_query

    # =========================
    # 2) ensure semantic_cache exist
    # =========================
    if "semantic_cache" not in state:
        state["semantic_cache"] = []

    # =========================
    # 3) avoid duplicate (IMPORTANT)
    # =========================
    exists = any(
        item.get("semantic_query") == final_query
        and item.get("runbook_title") == title
        for item in state["semantic_cache"]
    )

    if not exists:
        state["semantic_cache"].append({
            "semantic_query": final_query,
            "runbook_title": title,
            "service": service
        })

    # limit size (keep như cũ)
    state["semantic_cache"] = state["semantic_cache"][-5:]

    # =========================
    # 4) 🔥 update runtime cache
    # =========================
    add_to_runtime_semantic_cache(final_query, rb)

def find_runbook_by_title_service(title, service):
    """
    Map lại runbook từ data source (runbook JSON)
    """

    for rb in RUNBOOK_DATA:   # 🔥 thay đúng biến của anh
        if (
            rb.get("title") == title
            and rb.get("service") == service
        ):
            return rb

    return None


def rebuild_runtime_semantic_cache_from_sessions(sessions):
    """
    Rebuild FAISS semantic cache từ session.json
    """

    global SEMANTIC_CACHE_VECTORS
    global SEMANTIC_CACHE_ENTRIES
    global SEMANTIC_CACHE_INDEX

    # 1) reset runtime cache
    # ======================
    SEMANTIC_CACHE_VECTORS = []
    SEMANTIC_CACHE_ENTRIES = []
    SEMANTIC_CACHE_INDEX = None

    if not sessions:
        return

    # 2) gom semantic_cache từ tất cả session
    # =======================================
    for session_id, state in sessions.items():
        cache_items = state.get("semantic_cache", [])

        for item in cache_items:
            semantic_query = item.get("semantic_query")
            title = item.get("runbook_title")
            service = item.get("service")

            if not semantic_query or not title:
                continue

            # 3) tìm lại runbook đầy đủ
            # =========================
            rb = find_runbook_by_title_service(title, service)

            if not rb:
                continue

            # 4) tránh duplicate
            # =========================
            exists = any(
                entry.get("semantic_query") == semantic_query
                and entry.get("runbook_title") == title
                for entry in SEMANTIC_CACHE_ENTRIES
            )

            if exists:
                continue

            # 5) encode query
            # =========================
            vector = CACHE_MODEL.encode(
                [semantic_query],
                normalize_embeddings=True
            )[0]

            vector = np.array(vector, dtype="float32")

            SEMANTIC_CACHE_VECTORS.append(vector)

            SEMANTIC_CACHE_ENTRIES.append({
                "semantic_query": semantic_query,
                "runbook_title": title,
                "service": service,
                "runbook": rb,
                "source": "session_json"
            })

    # 6) build FAISS index
    # =========================
    rebuild_semantic_cache_index()
