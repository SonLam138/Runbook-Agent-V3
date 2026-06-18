import json
import chromadb
from chromadb.config import Settings

# load runbook
with open("data/runbooks.json", "r", encoding="utf-8") as f:
    runbooks = json.load(f)

client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = client.get_or_create_collection("runbooks")

for i, rb in enumerate(runbooks):
    text = f"{rb.get('service','')} {rb.get('keyword','')} {rb.get('description','')}"

    collection.add(
        documents=[text],
        metadatas=[rb],
        ids=[str(i)]
    )

print("✅ Loaded all runbooks into Chroma")