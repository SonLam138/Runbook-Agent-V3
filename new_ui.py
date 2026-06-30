import streamlit as st
import uuid

from app.load_chroma import load_chroma
from app.vector_store import ChromaStore

from app.agent import run_agent
from app.semantic_cache import save_feedback
from app.session_store import reset_session
from app.semantic_cache import RUNBOOK_DATA


def get_runbook_by_id(title):
    if not title:
        return None

    for rb in RUNBOOK_DATA:
        if str(rb.get("title")).strip().lower() == str(title).strip().lower():
            return rb

    return None


st.set_page_config(layout="wide")
st.title("🤖 IT Runbook Agent")


@st.cache_resource
def get_vector_store():
    chroma_collection = load_chroma()
    return ChromaStore(collection=chroma_collection)

VECTOR_STORE = get_vector_store()

# ====================
# INIT
# ====================
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.current_chat = cid
    st.session_state.chats[cid] = {
        "session_id": str(uuid.uuid4()),
        "messages": []
    }

# ✅ NEW: index cho Tier1 navigation
if "candidate_index" not in st.session_state:
    st.session_state["candidate_index"] = 0

# ====================
# SIDEBAR
# ====================
with st.sidebar:
    st.header("💬 Chats")

    if st.button("➕ New Chat"):
        cid = str(uuid.uuid4())
        st.session_state.current_chat = cid
        st.session_state.chats[cid] = {
            "session_id": str(uuid.uuid4()),
            "messages": []
        }
        st.session_state["candidate_index"] = 0
        st.rerun()

    for cid in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([4,1])

        with col1:
            if st.button(f"Chat {cid[:6]}", key=cid):
                st.session_state.current_chat = cid
                st.session_state["candidate_index"] = 0
                #st.rerun()

        with col2:
            if st.button("❌", key=f"del_{cid}"):
                del st.session_state.chats[cid]

                if st.session_state.current_chat == cid:
                    if st.session_state.chats:
                        st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                    else:
                        new_id = str(uuid.uuid4())
                        st.session_state.current_chat = new_id
                        st.session_state.chats[new_id] = {
                            "session_id": str(uuid.uuid4()),
                            "messages": []
                        }

                st.session_state["candidate_index"] = 0
                st.rerun()

# ====================
# CURRENT CHAT
# ====================
chat = st.session_state.chats[st.session_state.current_chat]
SESSION_ID = chat["session_id"]
messages = chat["messages"]

# ====================
# RENDER CHAT
# ====================
for i, msg in enumerate(messages):

    if msg["role"] == "user":
        st.markdown(f"👤 **{msg['text']}**")

    else:
        result = msg.get("result")

        # ✅ ===== PHASE 3 + 4: CANDIDATE CONTRACT =====
        if isinstance(result, dict) and result.get("status") == "candidate":

            data = result.get("data", {})
            tier1 = data.get("tier1", [])
            primary = data.get("primary")
            idx = st.session_state.get("candidate_index", 0)

            # ✅ safety index
            if idx >= len(tier1):
                idx = 0
                st.session_state["candidate_index"] = 0

            if tier1:

                # =========================
                # ✅ PRIMARY (Tier1[0])
                # =========================
                if primary:
                    st.markdown("### ✅ Hướng chính đề xuất")
                    st.markdown(f"📘 **{primary.get('short_label')}**")

                    rb = get_runbook_by_id(primary.get("short_label"))

                    # ✅ render full runbook
                    if rb:
                        if rb.get("precheck"):
                            st.markdown("### ✅ Precheck")
                            for p in rb["precheck"]:
                                st.markdown(f"- {p}")

                        if rb.get("steps"):
                            st.markdown("### 🔧 Steps")
                            for step in rb["steps"]:
                                st.markdown(f"- {step}")

                        if rb.get("postcheck"):
                            st.markdown("### 🔍 Postcheck")
                            for p in rb["postcheck"]:
                                st.markdown(f"- {p}")

                # =========================
                # ✅ NAVIGATION (idx > 0)
                # =========================
                if idx > 0 and idx < len(tier1):

                    c = tier1[idx]

                    st.markdown("### 🔄 Phương án khác trong cùng hướng")
                    st.markdown(f"📘 **{c.get('short_label')}**")

                    rb = get_runbook_by_id(c.get("short_label"))

                    if rb:
                        if rb.get("precheck"):
                            st.markdown("### ✅ Precheck")
                            for p in rb["precheck"]:
                                st.markdown(f"- {p}")

                        if rb.get("steps"):
                            st.markdown("### 🔧 Steps")
                            for step in rb["steps"]:
                                st.markdown(f"- {step}")

                        if rb.get("postcheck"):
                            st.markdown("### 🔍 Postcheck")
                            for p in rb["postcheck"]:
                                st.markdown(f"- {p}")

                # =========================
                # ✅ NEXT BUTTON
                # =========================
                col1, col2 = st.columns([1,3])

                with col1:
                    if st.button("➡️ Next", key=f"next_{i}"):
                        st.session_state["candidate_index"] = idx + 1

                # =========================
                # ✅ STATUS HELPER
                # =========================
                with col2:
                    if idx == 0:
                        st.caption(f"✅ Hướng chính (1/{len(tier1)})")
                    else:
                        st.caption(f"🔄 Phương án {idx+1}/{len(tier1)}")

            else:
                st.markdown("⚠️ Không có candidate phù hợp")

# ====================
# INPUT
# ====================
st.markdown("---")

user_input = st.text_input("Nhập câu hỏi...")

if st.button("📨 Gửi") and user_input.strip():

    # ✅ RESET index khi có câu hỏi mới
    st.session_state["candidate_index"] = 0

    messages.append({
        "role": "user",
        "text": user_input
    })

    with st.spinner("🤖 Agent đang tìm kiếm runbook..."):
        result = run_agent(SESSION_ID, user_input, VECTOR_STORE)

    messages.append({
        "role": "assistant",
        "result": result
    })

    st.rerun()