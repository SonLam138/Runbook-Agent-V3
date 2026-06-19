from script.RB_query_faiss import faiss_search_topk

RESULT_THRESHOLD = 0.50

def search_RB_topk(query, exclude_titles=None, topk=3, result_threshold=RESULT_THRESHOLD):
    """
    Wrapper cho FAISS top-k full runbook.
    """
    return faiss_search_topk(
        query=query,
        exclude_titles=exclude_titles or [],
        topk=topk,
        result_threshold=result_threshold
    )


def search_RB(query, result_threshold=RESULT_THRESHOLD):
    """
    Lấy 1 runbook tốt nhất.
    """
    results = search_RB_topk(
        query=query,
        exclude_titles=[],
        topk=1,
        result_threshold=result_threshold
    )

    if not results:
        return None

    return results[0]


def retrieve_candidates_meta(query, state, topk=3):
    """
    Candidate retrieval cho decision layer:
    - dùng ChromaDB
    - return full_candidates + meta_candidates (giữ nguyên contract cũ)
    """

    exclude_titles = state.get("tried_runbooks", [])
"""
#sửa khi thay faiss bằng ChromaDB
    # ✅ CHROMA QUERY
    results = chroma_collection.query(
        query_texts=[query],
        n_results=topk
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    full_candidates = []
    meta_candidates = []

    for i in range(len(docs)):
        distance = distances[i]
        score = 1 - distance   # ✅ bắt buộc

        meta = metadatas[i]

        title = meta.get("title", "")

        # ✅ exclude những cái đã thử
        if title in exclude_titles:
            continue

        rb = {
            "title": title,
            "service": meta.get("service", ""),
            "keyword": meta.get("keyword", ""),
            "description": meta.get("description", ""),
            "steps": meta.get("steps", ""),
            "score": score
        }

        full_candidates.append(rb)

        meta_candidates.append({
            "idx": len(full_candidates),
            "title": title,
            "service": rb["service"],
            "keyword": rb["keyword"],
            "description": rb["description"],
            "score": score
        })

    # ✅ SORT
    full_candidates = sorted(full_candidates, key=lambda x: x["score"], reverse=True)
    meta_candidates = sorted(meta_candidates, key=lambda x: x["score"], reverse=True)

    return full_candidates, meta_candidates
"""
def strong_match(full_candidates):
    if not full_candidates:
        return False

    best_score = full_candidates[0]["score"]

    if len(full_candidates) == 1:
        return best_score >= 0.60

    second_score = full_candidates[1]["score"]
    margin = best_score - second_score

    if best_score >= 0.60 and margin >= 0.03:
        return True

    if best_score >= 0.55 and margin >= 0.05:
        return True

    return False