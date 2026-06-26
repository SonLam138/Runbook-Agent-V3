# app/service_resolver_v2.py

import json
import re
from pathlib import Path

import numpy as np
import faiss
import requests

from app.load_model import embedding_model


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SERVICE_CATALOG_FILE = DATA_DIR / "knowledge_catalog.json"

# Rule thresholds
RULE_ACCEPT_THRESHOLD = 0.60
RULE_STRONG_THRESHOLD = 0.90

# Semantic thresholds
SEMANTIC_ACCEPT_THRESHOLD = 0.55
SEMANTIC_STRONG_THRESHOLD = 0.72

# Hybrid / decision thresholds
HYBRID_STRONG_THRESHOLD = 0.78
AMBIGUOUS_MARGIN = 0.08

# LLM fallback
LLM_ACCEPT_THRESHOLD = 0.75
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


# =====================================================
# BASIC UTILS
# =====================================================
def normalize(text: str) -> str:
    """
    Normalize text for rule matching.
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    if not text:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Service catalog file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


# =====================================================
# CATALOG ADAPTER
# =====================================================
def normalize_catalog_item(item):
    service = (
        item.get("service")
        or item.get("service_name")
        or item.get("service_id")
        or ""
    )

    aliases = item.get("aliases", [])

    keywords = (
        item.get("keywords")
        or item.get("common_keywords")
        or []
    )

    symptoms = item.get("symptoms", [])
    description = item.get("description", "")

    return {
        "service": service,
        "aliases": aliases,
        "keywords": keywords,
        "symptoms": symptoms,
        "description": description,
        "raw": item,
    }


def normalize_catalog(raw_catalog):
    """
    Support:
    1. service_catalog.json là list:
       [ {...}, {...} ]

    2. service_catalog.json là object:
       {
         "services": [...],
         "service_catalog": [...]
       }
    """

    if isinstance(raw_catalog, list):
        items = raw_catalog

    elif isinstance(raw_catalog, dict):
        if "service_catalog" in raw_catalog:
            items = raw_catalog.get("service_catalog", [])
        elif "services" in raw_catalog:
            items = raw_catalog.get("services", [])
        else:
            items = []
    else:
        items = []

    normalized = []

    for item in items:
        norm = normalize_catalog_item(item)
        if norm["service"]:
            normalized.append(norm)

    return normalized


# =====================================================
# LOAD SERVICE CATALOG
# =====================================================
RAW_SERVICE_CATALOG = load_json(SERVICE_CATALOG_FILE)
SERVICE_ITEMS = normalize_catalog(RAW_SERVICE_CATALOG)

if not SERVICE_ITEMS:
    raise ValueError("SERVICE_ITEMS is empty. Check service_catalog.json schema/content.")


# =====================================================
# BUILD SERVICE EMBEDDING INDEX
# =====================================================
def build_service_text(item: dict) -> str:
    service = item.get("service", "")
    aliases = ", ".join(item.get("aliases", []))
    keywords = ", ".join(item.get("keywords", []))
    symptoms = ", ".join(item.get("symptoms", []))
    description = item.get("description", "")

    return f"""
Service: {service}
Aliases: {aliases}
Keywords: {keywords}
Symptoms: {symptoms}
Description: {description}
""".strip()


SERVICE_TEXTS = [build_service_text(item) for item in SERVICE_ITEMS]

SERVICE_EMBEDDINGS = embedding_model.encode(
    SERVICE_TEXTS,
    normalize_embeddings=True
)

SERVICE_EMBEDDINGS = np.array(SERVICE_EMBEDDINGS, dtype="float32")

SERVICE_INDEX = faiss.IndexFlatIP(SERVICE_EMBEDDINGS.shape[1])
SERVICE_INDEX.add(SERVICE_EMBEDDINGS)


# =====================================================
# RULE MATCHING
# =====================================================
def match_term_score(query_norm: str, term: str, weight: float) -> float:
    term_norm = normalize(term)

    if not query_norm or not term_norm:
        return 0.0

    query_pad = f" {query_norm} "
    term_pad = f" {term_norm} "

    # ✅ Exact match
    if query_norm == term_norm:
        return min(1.0, weight + 0.08)

    # ✅ Exact phrase
    if term_pad in query_pad:
        return weight

    # ✅ Bag-of-words match (FIX QUAN TRỌNG)
    words = term_norm.split()
    if len(words) >= 2:
        match_count = sum(1 for w in words if w in query_norm)
        if match_count >= len(words) - 1:
            return weight - 0.15

    # ✅ Contains match (nhẹ hơn)
    if len(term_norm) >= 6 and term_norm in query_norm:
        return max(weight - 0.15, 0.0)

    return 0.0



def rule_match_candidates(query: str):
    """
    Return multi candidates từ rule matching.

    Output:
    [
      {
        "service": "...",
        "score": 0.92,
        "source": "rule",
        "matched_terms": [...]
      }
    ]
    """

    q = normalize(query)
    candidates = []

    for item in SERVICE_ITEMS:
        service = item.get("service", "")
        aliases = item.get("aliases", [])
        keywords = item.get("keywords", [])
        symptoms = item.get("symptoms", [])

        best_score = 0.0
        matched_terms = []

        # 1. keywords: mạnh nhất
        for term in keywords:
            score = match_term_score(q, term, 0.92)
            if score > 0:
                best_score = max(best_score, score)
                matched_terms.append(term)

        # 2. symptoms: rất quan trọng cho natural language/mail
        for term in symptoms:
            score = match_term_score(q, term, 0.88)
            if score > 0:
                best_score = max(best_score, score)
                matched_terms.append(term)

        # 3. aliases: trung bình
        for term in aliases:
            score = match_term_score(q, term, 0.82)
            if score > 0:
                best_score = max(best_score, score)
                matched_terms.append(term)

        # 4. service name: yếu hơn
        service_score = match_term_score(q, service, 0.72)
        if service_score > 0:
            best_score = max(best_score, service_score)
            matched_terms.append(service)

        if best_score >= RULE_ACCEPT_THRESHOLD:
            candidates.append({
                "service": service,
                "score": round(float(best_score), 4),
                "source": "rule",
                "matched_terms": sorted(list(set(matched_terms))),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# =====================================================
# SEMANTIC MATCHING
# =====================================================
def semantic_match_candidates(query: str, top_k: int = 5):
    """
    Semantic service matching bằng embedding + FAISS.
    Lấy top-k thay vì top-1 để detect ambiguity.
    """

    if not query:
        return []

    top_k = min(top_k, len(SERVICE_ITEMS))

    q_vec = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )
    q_vec = np.array(q_vec, dtype="float32")

    scores, indices = SERVICE_INDEX.search(q_vec, top_k)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)

        if idx == -1:
            continue

        score = float(score)

        if score < SEMANTIC_ACCEPT_THRESHOLD:
            continue

        item = SERVICE_ITEMS[idx]

        candidates.append({
            "service": item.get("service", ""),
            "score": round(score, 4),
            "source": "semantic",
            "matched_terms": [],
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# =====================================================
# MERGE CANDIDATES
# =====================================================
def merge_candidates(rule_candidates, semantic_candidates):
    """
    Merge rule + semantic.

    Nếu cùng service xuất hiện ở cả rule và semantic:
    - source = hybrid
    - final score được boost nhẹ
    """

    merged = {}

    for c in rule_candidates + semantic_candidates:
        service = c.get("service", "")

        if not service:
            continue

        if service not in merged:
            merged[service] = {
                "service": service,
                "rule_score": 0.0,
                "semantic_score": 0.0,
                "sources": set(),
                "matched_terms": [],
            }

        source = c.get("source")

        if source == "rule":
            merged[service]["rule_score"] = max(
                merged[service]["rule_score"],
                float(c.get("score", 0.0))
            )

        elif source == "semantic":
            merged[service]["semantic_score"] = max(
                merged[service]["semantic_score"],
                float(c.get("score", 0.0))
            )

        merged[service]["sources"].add(source)
        merged[service]["matched_terms"].extend(c.get("matched_terms", []))

    results = []

    for service, data in merged.items():
        rule_score = data["rule_score"]
        semantic_score = data["semantic_score"]

        if rule_score > 0 and semantic_score > 0:
            final_score = max(rule_score, semantic_score) + 0.08
            source = "hybrid"
        elif rule_score > 0:
            final_score = rule_score
            source = "rule"
        else:
            final_score = semantic_score
            source = "semantic"

        final_score = min(final_score, 1.0)

        results.append({
            "service": service,
            "score": round(float(final_score), 4),
            "source": source,
            "rule_score": round(float(rule_score), 4),
            "semantic_score": round(float(semantic_score), 4),
            "matched_terms": sorted(list(set(data["matched_terms"]))),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# =====================================================
# DECISION LAYER
# =====================================================
def make_result(
    service=None,
    confidence=None,
    source="none",
    is_strong=False,
    status="unresolved",
    candidates=None,
    reason="",
    trace=None,
):
    return {
        "service": service,
        "confidence": confidence,
        "score": confidence,
        "source": source,
        "is_strong": bool(is_strong),
        "status": status,
        "candidates": candidates or [],
        "reason": reason,
        "trace": trace or [],
    }


def decide_service(candidates, trace=None):
    """
    Chỉ decision layer mới được quyết định resolved/ambiguous/unresolved.
    Match layer không tự quyết.
    """

    trace = trace or []

    if not candidates:
        return make_result(
            status="unresolved",
            reason="no_candidates",
            candidates=[],
            trace=trace,
        )

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    top_score = float(top.get("score", 0.0))
    second_score = float(second.get("score", 0.0)) if second else 0.0
    margin = top_score - second_score

    trace.append({
        "stage": "decision_input",
        "top_service": top.get("service"),
        "top_score": round(top_score, 4),
        "second_service": second.get("service") if second else None,
        "second_score": round(second_score, 4),
        "margin": round(margin, 4),
        "top_source": top.get("source"),
    })

    # Strong hybrid
    if (
        top.get("source") == "hybrid"
        and top_score >= HYBRID_STRONG_THRESHOLD
        and margin >= AMBIGUOUS_MARGIN
    ):
        return make_result(
            service=top.get("service"),
            confidence=round(top_score, 4),
            source="hybrid",
            is_strong=True,
            status="resolved",
            candidates=candidates,
            reason="hybrid_strong_match",
            trace=trace,
        )

    # Strong rule
    if (
        top.get("source") == "rule"
        and top_score >= RULE_STRONG_THRESHOLD
        and margin >= 0.03
    ):
        return make_result(
            service=top.get("service"),
            confidence=round(top_score, 4),
            source="rule",
            is_strong=True,
            status="resolved",
            candidates=candidates,
            reason="rule_strong_match",
            trace=trace,
        )

    # Strong semantic
    if (
        top.get("source") == "semantic"
        and top_score >= SEMANTIC_STRONG_THRESHOLD
        and margin >= AMBIGUOUS_MARGIN
    ):
        return make_result(
            service=top.get("service"),
            confidence=round(top_score, 4),
            source="semantic",
            is_strong=True,
            status="resolved",
            candidates=candidates,
            reason="semantic_strong_match",
            trace=trace,
        )

    # Ambiguous: top candidates close
    if second and margin < AMBIGUOUS_MARGIN:
        return make_result(
            service=None,
            confidence=round(top_score, 4),
            source=top.get("source", "unknown"),
            is_strong=False,
            status="ambiguous",
            candidates=candidates,
            reason="top_candidates_too_close",
            trace=trace,
        )

    # Weak top
    return make_result(
        service=None,
        confidence=round(top_score, 4),
        source=top.get("source", "unknown"),
        is_strong=False,
        status="unresolved",
        candidates=candidates,
        reason="top_candidate_not_confident",
        trace=trace,
    )


# =====================================================
# LLM FALLBACK
# =====================================================
def clean_llm_json(text: str):
    """
    Extract JSON object from LLM response.
    """

    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except Exception:
        return None


def call_llm(prompt: str):
    """
    Ollama call.
    Nếu V4 của anh đã có call_llm riêng, sau này có thể thay function này bằng import dùng chung.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"LLM call failed: {response.text}")

    return response.json().get("response", "")


def build_llm_resolver_prompt(query, candidates):
    service_names = [item["service"] for item in SERVICE_ITEMS]

    compact_candidates = [
        {
            "service": c.get("service"),
            "score": c.get("score"),
            "source": c.get("source"),
            "matched_terms": c.get("matched_terms", []),
        }
        for c in candidates[:5]
    ]

    return f"""
Bạn là service resolver cho hệ thống IT ServiceDesk.

Nhiệm vụ:
Từ nội dung người dùng, chọn service phù hợp nhất trong danh sách cho trước.

Quy tắc bắt buộc:
- Chỉ được chọn service có trong danh sách service hợp lệ.
- Nếu không đủ thông tin, trả service = null.
- Không tự tạo service mới.
- Không trả lời người dùng.
- Không đưa hướng dẫn xử lý lỗi.
- Output phải là JSON duy nhất.

Danh sách service hợp lệ:
{json.dumps(service_names, ensure_ascii=False, indent=2)}

Nội dung người dùng:
{query}

Top candidates từ rule/semantic:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

Output JSON:
{{
  "service": null,
  "confidence": 0.0,
  "reason": ""
}}
""".strip()


def validate_llm_result(llm_json):
    if not isinstance(llm_json, dict):
        return None

    service = llm_json.get("service")
    confidence = llm_json.get("confidence", 0.0)
    reason = llm_json.get("reason", "")

    if service is None or service == "":
        return None

    valid_services = {item["service"] for item in SERVICE_ITEMS}

    if service not in valid_services:
        return None

    try:
        confidence = float(confidence)
    except Exception:
        return None

    if confidence < LLM_ACCEPT_THRESHOLD:
        return None

    return {
        "service": service,
        "confidence": round(confidence, 4),
        "score": round(confidence, 4),
        "source": "llm",
        "is_strong": True,
        "status": "resolved",
        "reason": reason or "llm_fallback_resolved",
    }


def llm_resolve_service_fallback(query, candidates, trace=None):
    """
    LLM fallback chỉ chạy khi deterministic resolver unresolved/ambiguous.
    """

    trace = trace or []

    try:
        prompt = build_llm_resolver_prompt(query, candidates)
        raw_response = call_llm(prompt)
        parsed = clean_llm_json(raw_response)

        trace.append({
            "stage": "llm_fallback_raw",
            "raw_response": raw_response,
            "parsed": parsed,
        })

        validated = validate_llm_result(parsed)

        if not validated:
            trace.append({
                "stage": "llm_fallback_rejected",
                "reason": "invalid_or_low_confidence",
            })
            return None

        validated["candidates"] = candidates
        validated["trace"] = trace
        validated["reason"] = validated.get("reason") or "llm_fallback_resolved"

        return validated

    except Exception as e:
        trace.append({
            "stage": "llm_fallback_error",
            "error": str(e),
        })
        return None


# =====================================================
# PUBLIC API
# =====================================================
def resolve_service_v2(query: str, state=None, use_llm=True):
    """
    Hybrid service resolver V2.

    Flow:
    1. rule_match_candidates
    2. semantic_match_candidates
    3. merge_candidates
    4. decide_service
    5. LLM fallback nếu unresolved/ambiguous

    Return:
    {
      "service": str | None,
      "confidence": float | None,
      "source": "rule|semantic|hybrid|llm|none",
      "is_strong": bool,
      "status": "resolved|ambiguous|unresolved",
      "candidates": [...],
      "reason": "...",
      "trace": [...]
    }
    """

    q = normalize(query)

    trace = [{
        "stage": "service_resolve_start",
        "query": query,
        "normalized_query": q,
    }]

    if not q:
        return make_result(
            status="unresolved",
            reason="empty_query",
            trace=trace,
        )

    # 1. Rule candidates
    rule_candidates = rule_match_candidates(q)

    trace.append({
        "stage": "rule_match_done",
        "count": len(rule_candidates),
        "candidates": rule_candidates[:5],
    })

    # 2. Semantic candidates
    semantic_candidates = semantic_match_candidates(q, top_k=5)

    trace.append({
        "stage": "semantic_match_done",
        "count": len(semantic_candidates),
        "candidates": semantic_candidates[:5],
    })

    # 3. Merge candidates
    candidates = merge_candidates(rule_candidates, semantic_candidates)

    trace.append({
        "stage": "candidates_merged",
        "count": len(candidates),
        "candidates": candidates[:5],
    })

    # 4. Deterministic decision
    decision = decide_service(candidates, trace=trace)

    trace.append({
        "stage": "deterministic_decision_done",
        "status": decision.get("status"),
        "service": decision.get("service"),
        "confidence": decision.get("confidence"),
        "source": decision.get("source"),
        "reason": decision.get("reason"),
    })

    if decision.get("status") == "resolved":
        return decision

    # 5. LLM fallback cuối cùng
    if use_llm:
        trace.append({
            "stage": "llm_fallback_called",
            "previous_status": decision.get("status"),
            "previous_reason": decision.get("reason"),
        })

        llm_decision = llm_resolve_service_fallback(
            query=query,
            candidates=candidates,
            trace=trace,
        )

        if llm_decision and llm_decision.get("status") == "resolved":
            return llm_decision

    # 6. Return unresolved/ambiguous nếu LLM không resolve được
    return decision


# Backward-compatible alias nếu muốn thay nhanh trong agent
resolve_service = resolve_service_v2