import json
import chromadb
from app.load_model import embedding_model


# =====================================================
# CONFIG
# =====================================================
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "runbooks"
DATA_FILE = "data/runbook_data.json"


# =====================================================
# HELPERS
# =====================================================
def safe_str(value):
    return str(value) if value is not None else ""


def normalize_keyword(kw):
    if isinstance(kw, list):
        return " ".join(str(k).strip() for k in kw if k)
    return str(kw).strip() if kw else ""


def normalize_list_text(value):
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if x)
    return str(value).strip() if value else ""


def build_runbook_text(rb):
    kw_text = normalize_keyword(rb.get("keyword", ""))
    intents_text = normalize_list_text(rb.get("intents", []))
    precheck_text = normalize_list_text(rb.get("precheck", []))
    steps_text = normalize_list_text(rb.get("steps", []))
    postcheck_text = normalize_list_text(rb.get("postcheck", []))

    text = f"""
    Title: {rb.get("title", "")}
    Service: {rb.get("service", "")}
    Keyword: {kw_text}
    Intents: {intents_text}
    Description: {rb.get("description", "")}
    Precheck: {precheck_text}
    Steps: {steps_text}
    Postcheck: {postcheck_text}
    """.strip()

    return text


def build_metadata(rb):
    title = safe_str(rb.get("title")).strip()

    return {
        "runbook_id": title,
        "title": title,
        "service": safe_str(rb.get("service")),
        "keyword": normalize_keyword(rb.get("keyword", "")),
        "description": safe_str(rb.get("description")).strip()
    }


# =====================================================
# LOAD COLLECTION (RUNTIME)
# =====================================================
def load_chroma(collection_name=COLLECTION_NAME, db_path=CHROMA_PATH):

    class PreLoadedEmbeddingFunction:
        def __call__(self, input):
            return embedding_model.encode(input, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=PreLoadedEmbeddingFunction()
    )

    return collection


# =====================================================
# BUILD CHROMA (ONLY RUN MANUALLY)
# =====================================================
def build_chroma():

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        runbooks = json.load(f)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("🧹 Deleted old collection")
    except:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, rb in enumerate(runbooks):

        title = (rb.get("title") or "").strip()

        if not title:
            print(f"⚠️ Missing title at index {i}, skip")
            continue

        text = build_runbook_text(rb)
        metadata = build_metadata(rb)

        emb = embedding_model.encode(
            [text],
            normalize_embeddings=True
        )[0]

        ids.append(title)
        documents.append(text)
        metadatas.append(metadata)
        embeddings.append(emb.tolist())

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"✅ Loaded {len(ids)} runbooks into Chroma")


# =====================================================
# MAIN (ONLY WHEN RUN DIRECTLY)
# =====================================================
if __name__ == "__main__":
    build_chroma()