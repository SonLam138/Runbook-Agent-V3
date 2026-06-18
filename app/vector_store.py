class VectorStore:
    def search(self, query, exclude_titles=None, top_k=3):
        raise NotImplementedError

    def retrieve_candidates(self, query, state, top_k=3):
        raise NotImplementedError
    

class FaissStore(VectorStore):
    def __init__(self, search_func, retrieve_func):
        self.search_func = search_func
        self.retrieve_func = retrieve_func

    def search(self, query, exclude_titles=None, top_k=3):
        return self.search_func(
            query=query,
            exclude_titles=exclude_titles or [],
            topk=top_k
        )

    def retrieve_candidates(self, query, state, top_k=3):
        return self.retrieve_func(
            query=query,
            state=state,
            topk=top_k
        )
    

import chromadb
from chromadb.config import Settings


class ChromaStore(VectorStore):
    def __init__(self, collection_name="runbooks"):
        self.client = chromadb.Client(
            Settings(persist_directory="./chroma_db")
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def search(self, query, exclude_titles=None, top_k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs = results.get("metadatas", [[]])[0]

        return docs

    def retrieve_candidates(self, query, state, top_k=3):
        exclude_titles = state.get("tried_runbooks", [])

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs = results.get("metadatas", [[]])[0]

        full_candidates = []
        meta_candidates = []

        for i, rb in enumerate(docs, start=1):

            title = rb.get("title")

            if title in exclude_titles:
                continue

            full_candidates.append(rb)

            meta_candidates.append({
                "idx": i,
                "title": rb.get("title", ""),
                "service": rb.get("service", ""),
                "keyword": rb.get("keyword", ""),
                "description": rb.get("description", ""),
                "score": 1.0  # ⚠️ tạm placeholder
            })

        return full_candidates, meta_candidates