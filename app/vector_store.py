import chromadb
from chromadb.config import Settings

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


class ChromaStore(VectorStore):
    """
    Chroma-backed VectorStore.

    Contract với agent.py:
    - retrieve_candidates(query, state, top_k=3)
      return full_candidates, meta_candidates

    - search(query, exclude_titles=None, top_k=3)
      dùng cho retry flow, trả list full candidates

    Lưu ý:
    - Class này KHÔNG tự load PersistentClient.
    - Collection được load từ bên ngoài bằng load_chroma().
    """

    def __init__(self, collection):
        self.collection = collection

    def _distance_to_score(self, dist):
        """
        Convert Chroma distance -> similarity score.

        Theo kết quả test hiện tại:
        - clear match: khoảng 0.60 - 0.67
        - borderline: khoảng 0.55
        - weak: dưới 0.50

        Công thức đang dùng:
            score = 1 - distance
        """
        if dist is None:
            return 0.0

        try:
            return 1 - float(dist)
        except Exception:
            return 0.0

    def search(self, query, exclude_titles=None, top_k=3):
        """
        Dùng cho retry:
        - Search lại cùng semantic_query
        - Exclude các runbook đã trả
        - Trả full runbook candidates
        """
        exclude_titles = set(exclude_titles or [])

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3
        )

        print("\n====================")
        print("🔁 CHROMA SEARCH QUERY:", query)
        print("🔁 RAW DISTANCES:", results.get("distances"))
        print("====================\n")

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        candidates = []

        for metadata, dist in zip(metadatas, distances):
            rb = dict(metadata or {})

            # ✅ FIX: đảm bảo luôn có runbook_id
            if "runbook_id" not in rb:

                # Case metadata đã có
                if isinstance(metadata, dict):
                    rb["runbook_id"] = metadata.get("runbook_id")

                # fallback: dùng title
                if not rb.get("runbook_id"):
                    rb["runbook_id"] = rb.get("title")

            # ❗ last fallback (hiếm)
            if not rb.get("runbook_id"):
                rb["runbook_id"] = str(rb.get("title", "unknown"))

            title = rb.get("title", "")

            if title in exclude_titles:
                print(f"⏭️ SKIP tried runbook: {title}")
                continue

            score = self._distance_to_score(dist)
            rb["score"] = score
            rb["distance"] = dist

            print(
                f"📊 RETRY RB: {title} | "
                f"dist={float(dist):.4f} | score={score:.4f}"
            )

            candidates.append(rb)

            if len(candidates) >= top_k:
                break

        candidates = sorted(
            candidates,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )

        print("\n🏆 RETRY TOP CANDIDATES:")
        for rb in candidates:
            print(f"➡️ {rb.get('title')} | score={rb.get('score', 0):.4f}")
        print("====================\n")

        return candidates

    def retrieve_candidates(self, query, state, top_k=3):
        """
        Dùng cho normal retrieval:
        - Search bằng Chroma
        - Convert distance -> score
        - Return full_candidates + meta_candidates
        """
        exclude_titles = set(state.get("tried_runbooks", []) or [])

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3
        )

        print("\n====================")
        print("🔎 CHROMA RETRIEVE QUERY:", query)
        print("🔎 RAW DISTANCES:", results.get("distances"))
        print("====================\n")

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        full_candidates = []

        for metadata, dist in zip(metadatas, distances):
            rb = dict(metadata or {})

            # ✅ FIX: đảm bảo luôn có runbook_id
            if "runbook_id" not in rb:

                # Case metadata đã có
                if isinstance(metadata, dict):
                    rb["runbook_id"] = metadata.get("runbook_id")

                # fallback: dùng title
                if not rb.get("runbook_id"):
                    rb["runbook_id"] = rb.get("title")

            # ❗ last fallback (hiếm)
            if not rb.get("runbook_id"):
                rb["runbook_id"] = str(rb.get("title", "unknown"))

            title = rb.get("title", "")

            if title in exclude_titles:
                print(f"⏭️ SKIP tried runbook: {title}")
                continue

            score = self._distance_to_score(dist)
            rb["score"] = score
            rb["distance"] = dist

            print(
                f"📊 RB: {title} | "
                f"dist={float(dist):.4f} | score={score:.4f}"
            )

            full_candidates.append(rb)

            if len(full_candidates) >= top_k:
                break

        full_candidates = sorted(
            full_candidates,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )

        full_candidates = full_candidates[:top_k]

        meta_candidates = []

        for i, rb in enumerate(full_candidates, start=1):
            meta_candidates.append({
                "idx": i,
                "title": rb.get("title", ""),
                "service": rb.get("service", ""),
                "keyword": rb.get("keyword", ""),
                "description": rb.get("description", ""),
                "score": rb.get("score", 0.0)
            })

        print("\n🏆 TOP CANDIDATES AFTER FILTER/SORT:")
        for rb in full_candidates:
            print(f"➡️ {rb.get('title')} | score={rb.get('score', 0):.4f}")
        print("====================\n")

        return full_candidates, meta_candidates