from app.load_chroma import load_chroma
from app.vector_store import ChromaStore

def init_vector_store():
    chroma_collection = load_chroma()
    return ChromaStore(collection=chroma_collection)