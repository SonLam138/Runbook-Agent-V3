import chromadb
from pathlib import Path
from app.load_model import embedding_model

# =====================================================
# CONFIG
# =====================================================
VECTOR_DB_DIR = Path(r"D:\vector_db")
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ✅ IMPORTANT: dùng PersistentClient
client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

# =====================================================
# COLLECTIONS
# =====================================================
runbook_collection = client.get_or_create_collection(
    name="runbooks"
)

memory_collection = client.get_or_create_collection(
    name="semantic_memory"
)

# =====================================================
# RUNBOOK FUNCTIONS
# =====================================================

def build_runbook_text(rb):
    return f"""
Runbook: {rb.get("title","")}
Service: {rb.get("service","")}

Keyword:
{rb.get("keyword","")}

Description:
{rb.get("description","")}
""".strip()


def add_runbooks(runbooks):
    texts = []
    metadatas = []
    ids = []

    for rb in runbooks:
        text = build_runbook_text(rb)
        texts.append(text)

        metadatas.append({
            "title": rb.get("title"),
            "service": rb.get("service"),
            "keyword": rb.get("keyword"),
            "source_file": rb.get("source_file")
        })

        # ✅ FIX: ID stable
        ids.append(rb.get("source_file"))

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    runbook_collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def rebuild_runbooks(runbooks):
    print("🧹 Clearing existing runbooks...")
    runbook_collection.delete(where={})  # delete all

    print("🚀 Rebuilding runbooks...")
    add_runbooks(runbooks)


def search_runbooks(query, topk=3):
    q_vec = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = runbook_collection.query(
        query_embeddings=q_vec,
        n_results=topk
    )

    return results


# =====================================================
# SEMANTIC MEMORY
# =====================================================

def save_memory(query, runbook_id, metadata=None):
    if metadata is None:
        metadata = {}

    vec = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    memory_collection.add(
        embeddings=vec,
        metadatas=[{
            "query": query,
            "runbook_id": runbook_id,
            **metadata
        }],
        ids=[f"mem_{hash(query)}"]  # ✅ stable enough cho memory
    )


def search_memory(query, threshold=0.82):
    q_vec = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = memory_collection.query(
        query_embeddings=q_vec,
        n_results=1
    )

    if not results["distances"]:
        return None

    score = 1 - results["distances"][0][0]

    if score < threshold:
        return None

    return {
        "metadata": results["metadatas"][0][0],
        "score": score
    }