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
def normalize_keyword(kw):
    if isinstance(kw, list):
        return " ".join(str(k).strip() for k in kw if k)
    return str(kw).strip() if kw else ""


def normalize_list_text(value):
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if x)
    return str(value).strip() if value else ""


def build_runbook_text(rb):
    """
    RICH TEXT cho embedding – IMPORTANT
    """
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
    kw_text = normalize_keyword(rb.get("keyword", ""))

    return {
        "title": rb.get("title", ""),
        "service": rb.get("service", ""),
        "keyword": kw_text,
        "description": rb.get("description", "")
    }

def load_chroma(collection_name="runbooks", db_path="./chroma_db"):
    """
    Dùng embedding_model đã load sẵn từ app.load_model
    để tránh load model 2 lần
    """
    # Tạo custom embedding function dùng model đã load
    class PreLoadedEmbeddingFunction:
        def __call__(self, input):
            # Chroma expects parameter named 'input', not 'texts'
            return embedding_model.encode(input, normalize_embeddings=True).tolist()
        
        def __repr__(self):
            return "PreLoadedEmbeddingFunction"
    
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=PreLoadedEmbeddingFunction()
    )

    return collection



# =====================================================
# LOAD DATA
# =====================================================
with open(DATA_FILE, "r", encoding="utf-8") as f:
    runbooks = json.load(f)


# =====================================================
# INIT CHROMA (PERSISTENT)
# =====================================================
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

# wipe collection cũ để tránh mix embedding
try:
    client.delete_collection(COLLECTION_NAME)
    print("🧹 Deleted old collection")
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# =====================================================
# INSERT DATA
# =====================================================
ids = []
documents = []
metadatas = []
embeddings = []

for i, rb in enumerate(runbooks):
    text = build_runbook_text(rb)
    metadata = build_metadata(rb)

    emb = embedding_model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    ids.append(str(i))
    documents.append(text)
    metadatas.append(metadata)
    embeddings.append(emb.tolist())


collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"✅ Loaded {len(runbooks)} runbooks into Chroma")