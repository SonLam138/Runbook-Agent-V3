from Log.logger import log


STRONG_THRESHOLD = 0.6


# =====================================================
# BUILD QUERY VIEWS
# =====================================================
def build_query_views(user_input, state, resolved_service):
    views = []

    # raw
    views.append({
        "type": "raw",
        "query": user_input,
        "weight": 1.0
    })

    # semantic
    semantic = state.get("semantic_query")
    if semantic and semantic != user_input:
        views.append({
            "type": "semantic",
            "query": semantic,
            "weight": 1.15
        })

    # service + issue
    service_name = ""

    if isinstance(resolved_service, dict):
        service_name = (
            resolved_service.get("service_name")
            or resolved_service.get("service_id")
            or ""
        )

    elif isinstance(resolved_service, str):
        service_name = resolved_service

    issue = state.get("issue_type", "")
    error = state.get("error_message", "")

    issue_part = issue or user_input

    service_query = " ".join(
        [service_name, issue_part, error]
    ).strip()


    if service_query:
        views.append({
            "type": "service_issue",
            "query": service_query,
            "weight": 1.25
        })

    log("multiquery_built", {
        "user_input": user_input,
        "semantic_query": state.get("semantic_query"),
        "resolved_service": resolved_service,
        "views": views
    })

    return views


# =====================================================
# MULTI QUERY RETRIEVE
# =====================================================
def retrieve_multi_query(views, vector_store, state, top_k=5):
    all_results = []

    for v in views:
        full_candidates, meta_candidates = vector_store.retrieve_candidates(
            query=v["query"],
            state=state,
            top_k=top_k
        )

        results = full_candidates  # ✅ FIX quan trọng

        log("multiquery_retrieval_per_query", {
            "query_type": v["type"],
            "query": v["query"],
            "top_results": [
                {
                    "runbook_id": r.get("runbook_id"),
                    "score": r.get("score")
                }
                for r in results[:3]
            ]
        })

        for r in results:

            # ✅ chuẩn hóa runbook_id
            if "runbook_id" not in r:
                if "metadata" in r and isinstance(r["metadata"], dict):
                    r["runbook_id"] = r["metadata"].get("runbook_id")
                else:
                    r["runbook_id"] = r.get("id")  # fallback

            if not r.get("runbook_id"):
                # bỏ candidate lỗi
                continue

            r["_query_type"] = v["type"]
            r["_query_weight"] = v["weight"]
            r["_base_score"] = r.get("score", 0)
            r["_weighted_score"] = r["_base_score"] * v["weight"]

            all_results.append(r)

    return all_results


# =====================================================
# MERGE + RERANK
# =====================================================
def merge_and_rerank(all_candidates):
    grouped = {}

    for c in all_candidates:
        rb_id = c["runbook_id"]

        if rb_id not in grouped:
            grouped[rb_id] = {
                "item": c,
                "scores": [],
                "query_types": set()
            }

        grouped[rb_id]["scores"].append(c["_weighted_score"])
        grouped[rb_id]["query_types"].add(c["_query_type"])

    merged = []

    for rb_id, g in grouped.items():
        item = g["item"]

        max_score = max(g["scores"])
        coverage = len(g["query_types"])

        coverage_bonus = min(0.05 * coverage, 0.15)

        final_score = max_score + coverage_bonus

        item["final_score"] = final_score
        item["coverage_count"] = coverage
        item["_query_types"] = list(g["query_types"])  # ✅ FIX

        merged.append(item)

    merged.sort(key=lambda x: x["final_score"], reverse=True)

    log("multiquery_merge_done", {
        "candidates": [
            {
                "runbook_id": c["runbook_id"],
                "final_score": c["final_score"],
                "coverage": c["coverage_count"]
            }
            for c in merged[:10]
        ]
    })

    return merged


# =====================================================
# ENRICH
# =====================================================

def find_runbook_record(runbook_data, rb_id):
    if not runbook_data:
        return {}

    # Case list (RUNBOOK_DATA)
    if isinstance(runbook_data, list):
        for item in runbook_data:
            if not isinstance(item, dict):
                continue

            title = (item.get("title") or "").strip()
            runbook_id = (item.get("runbook_id") or "").strip()

            # ✅ match theo title (identity hiện tại)
            if rb_id in [title, runbook_id]:
                return item

    # Case dict (runbook_map future)
    if isinstance(runbook_data, dict):
        return runbook_data.get(rb_id, {})

    return {}

def enrich_candidates(candidates, runbook_data, tier_name):
    enriched = []

    for i, c in enumerate(candidates):
        rb_id = c["runbook_id"]

        rb = find_runbook_record(runbook_data, rb_id)

        short_label = (
            rb.get("short_label")
            or rb.get("title")
            or c.get("title")
            or rb_id
        )

        llm_hint = (
            rb.get("llm_hint")
            or rb.get("description")
            or c.get("description")
            or ""
        )

        symptom_keywords = (
            rb.get("symptom_keywords")
            or rb.get("keyword")
            or c.get("keyword")
            or []
        )
        
        enriched.append({
            "runbook_id": rb_id,
            "service_id": (
                c.get("service_id")
                or c.get("service")
                or rb.get("service")
                or ""
            ),
            "rank": i + 1,
            "tier": c.get("_tier", None),
            "short_label": short_label,
            "llm_hint": llm_hint,
            "symptom_keywords": symptom_keywords,

            "scores": {
                "final_score": c["final_score"]
            },

            "match_signals": {
                "coverage_count": c["coverage_count"],
                "source_queries": c.get("_source_queries", []),
                "query_types": c.get("_query_types", [])
            }
        })

    log("candidate_enriched", {
        "sample": [
            {
                "runbook_id": c["runbook_id"],
                "short_label": c["short_label"],
                "llm_hint": c["llm_hint"][:80] if c["llm_hint"] else ""
            }
            for c in enriched[:3]
        ]
    })

    return enriched


# =====================================================
# SPLIT TIERS
# =====================================================
def split_tiers(candidates):

    if not candidates:
        return [], []

    def get_score(c):
        return (
            c.get("scores", {}).get("final_score")
            or c.get("final_score")
            or c.get("score")
            or 0.0
        )

    top_score = get_score(candidates[0])
    # ## Tìm những thằng gần top nhất để add vào tier1 - lưu ý không phải là strong ####
    tier1 = []
    for c in candidates:
        if get_score(c) >= top_score * 0.9:
            tier1.append(c)
        else:
            break

    strong_tier1 = [
        c for c in tier1
        if get_score(c) >= STRONG_THRESHOLD
    ]


    tier2 = candidates[len(tier1): len(tier1)+5]

    # ✅ PHẢI GIỮ
    for c in tier1:
        c["_tier"] = "tier1"

    for c in tier2:
        c["_tier"] = "tier2"

    for c in strong_tier1:
        c["_tier"] = "strong_tier1"

    log("candidate_split_tier", {
        "tier1": [c["runbook_id"] for c in tier1],
        "tier2": [c["runbook_id"] for c in tier2],
        "strong_tier1": [c["runbook_id"] for c in strong_tier1]
    })

    return tier1, tier2, strong_tier1

def group_tier2_into_lanes(tier2):

    lanes = {}

    for c in tier2:
        key = c.get("service_id") or "other"

        if key not in lanes:
            lanes[key] = {
                "lane_id": key,
                "lane_label": f"Hướng liên quan: {key}",
                "candidates": []
            }

        lanes[key]["candidates"].append(c)

    return list(lanes.values())


# =====================================================
# MAIN ENTRY
# =====================================================
def retrieve_candidates_multiquery(user_input, state, vector_store, runbook_data):

    resolved_service = state.get("resolved_service")

    # 1. build query
    views = build_query_views(user_input, state, resolved_service)

    # 2. retrieve ✅ FIX: truyền state
    raw = retrieve_multi_query(views, vector_store, state)

    # 3. merge
    merged = merge_and_rerank(raw)

    # 4. enrich
    #enriched = enrich_candidates(merged, runbook_data)

    # 5. split
    #tier1, tier2 = split_tiers(enriched)
    tier1, tier2, strong_tier1 = split_tiers(merged)

    #enrich
    tier1 = enrich_candidates(tier1, runbook_data, "tier1")
    tier2 = enrich_candidates(tier2, runbook_data, "tier2")
    strong_tier1 = enrich_candidates(strong_tier1, runbook_data, "tier1")

    # group tier2 lane
    tier2_lanes = group_tier2_into_lanes(tier2)

    return {
        "tier1_candidates": tier1,
        "tier2_lanes": tier2_lanes,
        "strong_tier1": strong_tier1
    }