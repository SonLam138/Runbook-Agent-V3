import streamlit as st
import uuid

from app.load_chroma import load_chroma
from app.vector_store import ChromaStore
from app.agent import run_agent
from app.semantic_cache import RUNBOOK_DATA


# ====================
# HELPERS
# ====================
def get_runbook_by_id(title):
    if not title:
        return None
    for rb in RUNBOOK_DATA:
        if str(rb.get("title")).strip().lower() == str(title).strip().lower():
            return rb
    return None


# ====================
# CONFIG
# ====================
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


# ====================
# SIDEBAR (KEEP)
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
        st.rerun()

    for cid in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button(f"Chat {cid[:6]}", key=cid):
                st.session_state.current_chat = cid

        with col2:
            if st.button("❌", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if not st.session_state.chats:
                    new_id = str(uuid.uuid4())
                    st.session_state.current_chat = new_id
                    st.session_state.chats[new_id] = {
                        "session_id": str(uuid.uuid4()),
                        "messages": []
                    }
                else:
                    st.session_state.current_chat = list(st.session_state.chats.keys())[0]
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
        st.chat_message("user").write(msg["text"])

    else:
        result = msg.get("result", {})
        if result is None:
            continue   # ✅ dừng tại đây, chờ rerun vào 8B.1

        with st.chat_message("assistant"):

            assistant_turn = result.get("assistant_turn", {})
            if result is None:
                st.stop()   # ✅ dừng tại đây, chờ rerun vào 8B.1

            data = result.get("data", {})

            # =========================
            # ✅ MESSAGE
            # =========================
            message = assistant_turn.get("message", "")
            if message:
                st.write(message)

            # =========================
            # ✅ PRIMARY RUNBOOK
            # =========================
            primary = data.get("primary")

            if primary:
                title = primary.get("short_label")
                st.markdown(f"### 📘 {title}")

                display_policy = assistant_turn.get("display_policy", {})

                if display_policy.get("show_full_runbook", False):
                    rb = get_runbook_by_id(title)
                    #rb = primary
                    if rb:
                        if rb.get("precheck"):
                            st.markdown("### ✅ Precheck")
                            for p in rb["precheck"]:
                                st.write("-", p)

                        if rb.get("steps"):
                            st.markdown("### 🔧 Steps")
                            for s in rb["steps"]:
                                st.write("-", s)

                        if rb.get("postcheck"):
                            st.markdown("### 🔍 Postcheck")
                            for p in rb["postcheck"]:
                                st.write("-", p)

            # =========================
            # ✅ ACTIONS (GENERIC)
            # =========================
            actions = assistant_turn.get("suggested_utterances", [])

            if actions:
                st.markdown("💡 Bạn có thể:")

                cols = st.columns(len(actions))

                for j, act in enumerate(actions):

                    if cols[j].button(
                        act.get("label", ""),
                        key=f"btn_{i}_{j}"
                    ):

                        utterance = act.get("utterance")

                        # ✅ push user message
                        messages.append({
                            "role": "user",
                            "text": utterance
                        })

                        # ✅ call agent
                        result2 = run_agent(
                            SESSION_ID,
                            utterance,
                            VECTOR_STORE
                        )

                        messages.append({
                            "role": "assistant",
                            "result": result2
                        })

                        st.rerun()


# ====================
# INPUT
# ====================
st.markdown("---")

# ✅ init state
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = None

if "processing" not in st.session_state:
    st.session_state.processing = False


# ✅ chat input
user_input = st.chat_input("Nhập yêu cầu...")


# ====================
# PROCESS INPUT
# ====================
if user_input:

    # ✅ CHẶN double processing trong cùng 1 rerun cycle
    if st.session_state.processing:
        st.stop()

    # ✅ CHẶN xử lý lại input cũ
    if user_input == st.session_state.last_processed_input:
        st.stop()

    # ✅ mark ngay từ đầu (QUAN TRỌNG)
    st.session_state.processing = True
    st.session_state.last_processed_input = user_input

    # ✅ append user
    messages.append({
        "role": "user",
        "text": user_input
    })

    # ✅ call agent (1 lần duy nhất)
    result = run_agent(
        SESSION_ID,
        user_input,
        VECTOR_STORE
    )

    # ✅ tránh append duplicate assistant
    if not messages or messages[-1].get("result") != result:
        messages.append({
            "role": "assistant",
            "result": result
        })

    # ✅ reset processing flag cho turn sau
    st.session_state.processing = False

    # ✅ rerun để render clean UI
    st.rerun()