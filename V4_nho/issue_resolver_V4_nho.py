import json
import re
import math
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from service_resolver_V4_nho import normalize_text, tokenize


# ========================
# PATH CONFIG
# ========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent   # đúng với project D:\Runbook_Agent

DATA_DIR = PROJECT_ROOT / "V4_nho"
RB_DIR = PROJECT_ROOT / "data"

KEYWORD_CATALOG_JSON = DATA_DIR / "keyword_catalog_V4_nho.json"
RUNBOOK_JSON = RB_DIR / "runbook_data.json"


# ========================
# CONFIG
# ========================

TOP_K = 5

# Threshold chính:
# - score < threshold  => weak
# - score >= threshold => đủ mạnh để xét resolve
RAW_SCORE_RESOLVE_THRESHOLD = 0.62

# Nếu top1 và top2 quá sát nhau thì coi là ambiguous
AMBIGUITY_MARGIN = 0.12

# Single-token phrase chỉ được tính nếu token đủ đặc trưng trong service scope
MIN_SINGLE_TOKEN_IDF = 1.65

# Multi-token phrase cần tối thiểu 2 token overlap,
# trừ khi exact phrase match nguyên cụm.
MIN_TOKEN_OVERLAP = 2

# Field priority theo thiết kế đã chốt
FIELD_WEIGHTS = {
    "keyword_list": 1.20,
    "intents": 1.00,
    "patterns": 0.85,
    "synonyms": 0.75,
    "title": 0.40
}

# Exact phrase bonus theo field
EXACT_BONUS_BY_FIELD = {
    "keyword_list": 0.35,
    "intents": 0.25,
    "patterns": 0.18,
    "synonyms": 0.15,
    "title": 0.08
}

# Dùng để normalize confidence về khoảng 0..1
# Max gần đúng: keyword_list exact = 1.20 + 0.35 = 1.55
MAX_EXPECTED_SCORE = 1.55

# LLM fallback tuning
LLM_MODEL = "mistral"
LLM_TIMEOUT = 12
LLM_NUM_PREDICT = 128

# Giới hạn text đưa vào LLM
LLM_EMAIL_LIMIT = 1200
LLM_TEXT_LIMIT = 220


# ========================
# BASIC IO
# ========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_keyword_entries(path=KEYWORD_CATALOG_JSON) -> List[Dict[str, Any]]:
    data = load_json(path)
    return data.get("entries", [])


def load_runbooks(path=RUNBOOK_JSON) -> List[Dict[str, Any]]:
    return load_json(path)


def build_runbook_index(runbooks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Index runbook_data theo title.
    title là khóa resolver trả về để lấy steps/precheck/postcheck.
    """
    index = {}

    for rb in runbooks:
        title = str(rb.get("title", "")).strip()

        if title:
            index[title] = rb

    return index


# ========================
# BASIC UTILS
# ========================

def as_list(value) -> List[str]:
    """
    Convert field về list string an toàn.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    value = str(value).strip()

    if not value:
        return []

    return [value]


def truncate_text(text: str, limit: int = LLM_TEXT_LIMIT) -> str:
    text = str(text or "").strip()

    if len(text) <= limit:
        return text

    return text[:limit].strip()


def get_entry_phrases(entry: Dict[str, Any], field: str) -> List[str]:
    """
    Lấy phrases của một field trong keyword_catalog entry.

    Field bình thường:
    - keyword_list
    - intents
    - patterns
    - synonyms

    Field đặc biệt:
    - title
    """
    if field == "title":
        title = str(entry.get("title", "")).strip()
        return [title] if title else []

    return as_list(entry.get(field, []))


def normalize_confidence(score: float) -> float:
    """
    Convert raw score sang confidence 0..1.
    """
    return round(min(1.0, score / MAX_EXPECTED_SCORE), 4)


def make_candidate_result(
    entry: Dict[str, Any],
    score: float,
    confidence: float,
    match_type: str,
    matched_field: str,
    matched_text: Optional[str] = None,
    overlap_tokens: Optional[List[str]] = None,
    coverage: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    result = {
        "title": entry.get("title"),
        "service": entry.get("service"),
        "service_id": entry.get("service_id"),
        "service_name": entry.get("service_name"),
        "technical_services": entry.get("technical_services", []),

        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "match_type": match_type,
        "matched_field": matched_field,
        "matched_text": matched_text,
        "overlap_tokens": overlap_tokens or [],
        "coverage": coverage,

        "priority": entry.get("priority", 1.0),
        "source_file": entry.get("source_file"),
        "source_hash": entry.get("source_hash")
    }

    if extra:
        result.update(extra)

    return result


# ========================
# SCOPE FILTER
# ========================

def filter_entries_by_service(
    entries: List[Dict[str, Any]],
    resolved_service_id: str
) -> List[Dict[str, Any]]:
    """
    V4_nhỏ chỉ search trong service đã được service_resolver resolve.
    Không search toàn keyword_catalog.
    """
    return [
        entry for entry in entries
        if entry.get("service_id") == resolved_service_id
    ]


# ========================
# TOKEN STATS
# ========================

def build_scope_token_stats(scoped_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build token statistics trong phạm vi service đã resolved.

    Ý nghĩa:
    - Token xuất hiện nhiều runbook trong cùng service scope thì yếu hơn.
    - Token hiếm hơn thì mạnh hơn.

    IDF:
    idf = log((N + 1) / (df + 1)) + 1
    """

    fields = [
        "keyword_list",
        "intents",
        "patterns",
        "synonyms",
        "title"
    ]

    doc_freq = {}
    total_docs = len(scoped_entries)

    for entry in scoped_entries:
        entry_tokens = set()

        for field in fields:
            phrases = get_entry_phrases(entry, field)

            for phrase in phrases:
                entry_tokens.update(tokenize(phrase))

        for token in entry_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    idf = {}

    for token, df in doc_freq.items():
        idf[token] = math.log((total_docs + 1) / (df + 1)) + 1

    return {
        "total_docs": total_docs,
        "doc_freq": doc_freq,
        "idf": idf
    }


def get_token_idf(token: str, token_stats: Dict[str, Any]) -> float:
    return token_stats.get("idf", {}).get(token, 1.0)


# ========================
# CORE SCORING
# ========================

def score_phrase(
    email_text: str,
    phrase: str,
    field: str,
    token_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Core scoring cho 1 phrase.

    Score dựa trên:
    - field_weight: keyword_list > intents > patterns > synonyms > title
    - weighted_coverage: coverage theo IDF trong service scope
    - evidence_strength: độ mạnh tổng của token overlap
    - exact_bonus: nếu phrase nằm nguyên trong email
    """

    clean_email = normalize_text(email_text)
    clean_phrase = normalize_text(phrase)

    if not clean_phrase:
        return {
            "matched": False,
            "reason": "empty_phrase"
        }

    email_tokens = tokenize(clean_email)
    phrase_tokens = tokenize(clean_phrase)

    if not phrase_tokens:
        return {
            "matched": False,
            "reason": "empty_phrase_tokens"
        }

    overlap = email_tokens.intersection(phrase_tokens)
    is_exact_phrase = clean_phrase in clean_email

    if not overlap:
        return {
            "matched": False,
            "reason": "no_overlap"
        }

    # Single-token phrase chỉ có giá trị nếu token đủ đặc trưng trong service scope.
    # Ví dụ "Exchange" thường quá phổ biến nên không nên chấm mạnh.
    if len(phrase_tokens) == 1:
        only_token = next(iter(phrase_tokens))
        only_token_idf = get_token_idf(only_token, token_stats)

        if only_token_idf < MIN_SINGLE_TOKEN_IDF:
            return {
                "matched": False,
                "reason": "single_token_too_common",
                "overlap_tokens": list(overlap),
                "token_idf": round(only_token_idf, 4)
            }

    # Multi-token phrase cần ít nhất 2 token overlap,
    # trừ trường hợp exact phrase match nguyên cụm.
    if len(phrase_tokens) >= 2:
        if len(overlap) < MIN_TOKEN_OVERLAP and not is_exact_phrase:
            return {
                "matched": False,
                "reason": "insufficient_overlap",
                "overlap_tokens": list(overlap),
                "overlap_count": len(overlap),
                "phrase_token_count": len(phrase_tokens)
            }

    phrase_idf_total = sum(
        get_token_idf(token, token_stats)
        for token in phrase_tokens
    )

    overlap_idf_total = sum(
        get_token_idf(token, token_stats)
        for token in overlap
    )

    if phrase_idf_total <= 0:
        return {
            "matched": False,
            "reason": "invalid_phrase_idf"
        }

    weighted_coverage = overlap_idf_total / phrase_idf_total

    # Evidence strength phản ánh tổng độ mạnh của overlap.
    # Token overlap càng đặc trưng thì càng mạnh.
    evidence_strength = min(1.0, overlap_idf_total / 3.0)

    field_weight = FIELD_WEIGHTS.get(field, 0.5)

    base_score = field_weight * (
        0.70 * weighted_coverage +
        0.30 * evidence_strength
    )

    exact_bonus = 0.0

    if is_exact_phrase:
        exact_bonus = EXACT_BONUS_BY_FIELD.get(field, 0.0)

    final_score = base_score + exact_bonus
    confidence = normalize_confidence(final_score)

    return {
        "matched": True,
        "score": round(final_score, 4),
        "confidence": confidence,
        "match_type": "exact_phrase" if is_exact_phrase else "token_overlap",
        "overlap_tokens": list(overlap),
        "coverage": round(weighted_coverage, 4),
        "is_exact_phrase": is_exact_phrase,
        "phrase_token_count": len(phrase_tokens),
        "overlap_count": len(overlap),
        "phrase_idf_total": round(phrase_idf_total, 4),
        "overlap_idf_total": round(overlap_idf_total, 4),
        "evidence_strength": round(evidence_strength, 4),
        "field_weight": field_weight,
        "exact_bonus": exact_bonus,
        "reason": "matched"
    }


def search_field(
    email_text: str,
    scoped_entries: List[Dict[str, Any]],
    field: str,
    token_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    ✅ PATCH: Entry-level ranking

    Mỗi entry chỉ có 1 result:
    → là best phrase của entry đó
    """

    entry_best = {}

    for entry in scoped_entries:
        priority = float(entry.get("priority", 1.0))
        entry_title = entry.get("title")

        best_result = None
        best_score = 0.0

        for phrase in get_entry_phrases(entry, field):
            scored = score_phrase(
                email_text=email_text,
                phrase=phrase,
                field=field,
                token_stats=token_stats
            )

            if not scored.get("matched"):
                continue

            raw_score = scored["score"] * priority

            if raw_score > best_score:
                best_score = raw_score

                best_result = make_candidate_result(
                    entry=entry,
                    score=raw_score,
                    confidence=normalize_confidence(raw_score),
                    match_type=scored.get("match_type"),
                    matched_field=field,
                    matched_text=phrase,
                    overlap_tokens=scored.get("overlap_tokens", []),
                    coverage=scored.get("coverage"),
                    extra={
                        "phrase_token_count": scored.get("phrase_token_count"),
                        "overlap_count": scored.get("overlap_count"),
                        "phrase_idf_total": scored.get("phrase_idf_total"),
                        "overlap_idf_total": scored.get("overlap_idf_total"),
                        "evidence_strength": scored.get("evidence_strength"),
                        "field_weight": scored.get("field_weight"),
                        "exact_bonus": scored.get("exact_bonus")
                    }
                )

        # ✅ chỉ giữ lại 1 result cho entry
        if best_result:
            entry_best[entry_title] = best_result

    results = sorted(
        entry_best.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ========================
# DECISION LOGIC
# ========================

def decide_ranked_results(
    results: List[Dict[str, Any]],
    decision_prefix: str
) -> Dict[str, Any]:
    """
    Quyết định strong / weak / ambiguous cho một tầng match.

    Rule:
    - no result            => no_match
    - best_score<threshold => weak
    - best và second sát   => ambiguous
    - còn lại              => resolved
    """

    if not results:
        return {
            "resolved": False,
            "decision": f"no_match_{decision_prefix}",
            "need_llm_fallback": False,
            "confidence": 0.0,
            "best_candidate": None,
            "top_k": []
        }

    best = results[0]
    second = results[1] if len(results) > 1 else None

    best_score = best["score"]
    best_confidence = best["confidence"]
    # ❗ KHÔNG cho keyword_list resolve nếu không phải exact match
    if decision_prefix == "keyword_list":
        if best.get("match_type") != "exact_phrase":
            return {
                "resolved": False,
                "decision": "weak_keyword_list_non_exact",
                "need_llm_fallback": False,
                "confidence": best_confidence,
                "best_candidate": best,
                "top_k": results[:TOP_K]
            }

        if best.get("phrase_token_count", 0) == 1 and best.get("coverage") < 1.0:
            return {
                "resolved": False,
                "decision": "weak_keyword_single_token",
            "need_llm_fallback": False,
                "confidence": best_confidence,
                "best_candidate": best,
                "top_k": results[:TOP_K]
            }

    # WEAK
    if best_score < RAW_SCORE_RESOLVE_THRESHOLD:
        return {
            "resolved": False,
            "decision": f"weak_{decision_prefix}",
            "need_llm_fallback": False,
            "confidence": best_confidence,
            "best_candidate": best,
            "top_k": results[:TOP_K]
        }

    # AMBIGUOUS
    if second:
        second_score = second["score"]

        if second_score >= RAW_SCORE_RESOLVE_THRESHOLD:
            gap = best_score - second_score
            gap_ratio = gap / max(best_score, 1e-6)

            if gap_ratio < AMBIGUITY_MARGIN:
                return {
                    "resolved": False,
                    "decision": f"ambiguous_{decision_prefix}",
                    "need_llm_fallback": False,
                    "confidence": best_confidence,
                    "best_candidate": best,
                    "top_k": results[:TOP_K]
                }

    # STRONG
    return {
        "resolved": True,
        "decision": f"resolved_by_{decision_prefix}",
        "need_llm_fallback": False,
        "confidence": best_confidence,
        "best_candidate": best,
        "top_k": results[:TOP_K]
    }


# ========================
# FIELD PRIORITY SEARCH
# ========================

def field_priority_search(
    email_text: str,
    scoped_entries: List[Dict[str, Any]],
    token_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Match theo thứ tự ưu tiên đã chốt:

    1. keyword_list
    2. intents
    3. patterns
    4. synonyms

    Mỗi tầng đều có:
    - no match
    - weak
    - ambiguous
    - resolved

    Nếu tầng hiện tại không resolved thì đi xuống tầng tiếp theo.
    """

    ordered_fields = [
        "keyword_list",
        "intents",
        "patterns",
        "synonyms"
    ]

    field_trace = []

    last_decision = None

    for field in ordered_fields:
        field_results = search_field(
            email_text=email_text,
            scoped_entries=scoped_entries,
            field=field,
            token_stats=token_stats
        )

        decision = decide_ranked_results(
            results=field_results,
            decision_prefix=field
        )

        field_trace.append({
            "field": field,
            "decision": decision.get("decision"),
            "confidence": decision.get("confidence"),
            "best_candidate": decision.get("best_candidate"),
            "top_k": decision.get("top_k", [])
        })

        last_decision = decision

        if decision.get("resolved"):
            decision["field_trace"] = field_trace
            return decision

    return {
        "resolved": False,
        "decision": "field_priority_not_resolved",
        "need_llm_fallback": False,
        "confidence": last_decision.get("confidence", 0.0) if last_decision else 0.0,
        "best_candidate": last_decision.get("best_candidate") if last_decision else None,
        "top_k": last_decision.get("top_k", []) if last_decision else [],
        "field_trace": field_trace
    }


# ========================
# GLOBAL TOKEN OVERLAP
# ========================

def global_token_overlap_search(
    email_text: str,
    scoped_entries: List[Dict[str, Any]],
    token_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Last deterministic layer.

    Sau khi field-priority không resolve được,
    resolver gom evidence trên tất cả field:

    - keyword_list
    - intents
    - patterns
    - synonyms
    - title

    Mỗi runbook chỉ lấy evidence mạnh nhất.
    """

    fields = [
        "keyword_list",
        "intents",
        "patterns",
        "synonyms",
        "title"
    ]

    entry_best = {}

    for field in fields:
        field_results = search_field(
            email_text=email_text,
            scoped_entries=scoped_entries,
            field=field,
            token_stats=token_stats
        )

        for result in field_results:
            title = result.get("title")

            if not title:
                continue

            existing = entry_best.get(title)

            if not existing or result["score"] > existing["score"]:
                entry_best[title] = result

    results = sorted(
        entry_best.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return decide_ranked_results(
        results=results,
        decision_prefix="global_token_overlap"
    )


# ========================
# LLM FALLBACK - OPTION 3
# ========================

def compact_entry_for_llm(entry: Dict[str, Any], candidate_id: int) -> Dict[str, Any]:
    """
    Compact scope để LLM xử lý nhanh hơn.

    Không đưa:
    - steps
    - precheck
    - postcheck
    - full raw runbook
    """

    return {
        "candidate_id": candidate_id,
        "title": entry.get("title"),
        "service": entry.get("service"),
        "keyword_list": [
            truncate_text(x, 120)
            for x in as_list(entry.get("keyword_list", []))
        ],
        "intents": [
            truncate_text(x, 180)
            for x in as_list(entry.get("intents", []))[:2]
        ],
        "patterns": [
            truncate_text(x, 120)
            for x in as_list(entry.get("patterns", []))[:5]
        ],
        "synonyms": [
            truncate_text(x, 120)
            for x in as_list(entry.get("synonyms", []))[:5]
        ]
    }


def build_llm_prompt(
    email_text: str,
    resolved_service_id: str,
    scoped_entries: List[Dict[str, Any]]
) -> str:
    """
    LLM chỉ được thấy issue scope củ
    """

    compact_scope = [
        compact_entry_for_llm(entry, idx)
        for idx, entry in enumerate(scoped_entries)
    ]

    return f"""
Bạn là IT ServiceDesk issue resolver.

Nhiệm vụ:
- Đọc email.
- Chọn đúng 1 runbook candidate phù hợp nhất trong ISSUE_SCOPE.
- Chỉ được chọn candidate_id có trong ISSUE_SCOPE.
- Không được tự tạo runbook mới.
- Nếu không candidate nào phù hợp, trả candidate_id = null.
- Trả JSON duy nhất, không giải thích dài.

Output JSON:
{{
  "candidate_id": null,
  "confidence": 0.0,
  "reason": "short reason"
}}

RESOLVED_SERVICE_ID:
{resolved_service_id}

EMAIL:
{truncate_text(email_text, LLM_EMAIL_LIMIT)}

ISSUE_SCOPE:
{json.dumps(compact_scope, ensure_ascii=False)}
"""


def clean_llm_json(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(response_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None


def call_llm_fallback(
    email_text: str,
    resolved_service_id: str,
    scoped_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Option 3:
    Chỉ gọi khi deterministic fail.

    Tối ưu tốc độ:
    - prompt compact
    - temperature=0
    - num_predict thấp
    - timeout ngắn
    """

    prompt = build_llm_prompt(
        email_text=email_text,
        resolved_service_id=resolved_service_id,
        scoped_entries=scoped_entries
    )

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": LLM_NUM_PREDICT
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=LLM_TIMEOUT
        )

        if response.status_code != 200:
            return {
                "resolved": False,
                "decision": "llm_call_failed",
                "need_llm_fallback": False,
                "confidence": 0.0,
                "error": response.text
            }

        raw_response = response.json().get("response", "")
        parsed = clean_llm_json(raw_response)

        if not parsed:
            return {
                "resolved": False,
                "decision": "llm_invalid_json",
                "need_llm_fallback": False,
                "confidence": 0.0,
                "raw_response": raw_response
            }

        candidate_id = parsed.get("candidate_id")
        confidence = float(parsed.get("confidence", 0.0) or 0.0)

        if candidate_id is None:
            return {
                "resolved": False,
                "decision": "llm_no_candidate",
                "need_llm_fallback": False,
                "confidence": round(confidence, 4),
                "llm_reason": parsed.get("reason")
            }

        if not isinstance(candidate_id, int):
            return {
                "resolved": False,
                "decision": "llm_invalid_candidate_id",
                "need_llm_fallback": False,
                "confidence": round(confidence, 4),
                "llm_reason": parsed.get("reason")
            }

        if candidate_id < 0 or candidate_id >= len(scoped_entries):
            return {
                "resolved": False,
                "decision": "llm_candidate_out_of_scope",
                "need_llm_fallback": False,
                "confidence": round(confidence, 4),
                "llm_reason": parsed.get("reason")
            }

        selected = scoped_entries[candidate_id]

        best_candidate = make_candidate_result(
            entry=selected,
            score=confidence,
            confidence=confidence,
            match_type="llm_fallback",
            matched_field="llm_scope",
            matched_text=parsed.get("reason"),
            overlap_tokens=[],
            coverage=None
        )

        return {
            "resolved": True,
            "decision": "resolved_by_llm_fallback",
            "need_llm_fallback": False,
            "confidence": round(confidence, 4),
            "best_candidate": best_candidate,
            "llm_reason": parsed.get("reason")
        }

    except Exception as e:
        return {
            "resolved": False,
            "decision": "llm_exception",
            "need_llm_fallback": False,
            "confidence": 0.0,
            "error": str(e)
        }


# ========================
# RUNBOOK ATTACHMENT
# ========================

def attach_runbook_content(
    result: Dict[str, Any],
    runbook_index: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Sau khi resolver chọn được title,
    lookup runbook_data.json để lấy nội dung thật.
    """

    if not result.get("resolved"):
        return result

    best = result.get("best_candidate") or {}
    title = best.get("title")

    if not title:
        result["runbook_found"] = False
        result["runbook_lookup_decision"] = "missing_title"
        return result

    rb = runbook_index.get(title)

    if not rb:
        result["runbook_found"] = False
        result["runbook_lookup_decision"] = "title_not_found_in_runbook_data"
        return result

    result["runbook_found"] = True
    result["runbook"] = {
        "title": rb.get("title"),
        "service": rb.get("service"),
        "keyword": rb.get("keyword"),
        "description": rb.get("description"),
        "precheck": rb.get("precheck", []),
        "steps": rb.get("steps", []),
        "postcheck": rb.get("postcheck", []),
        "source_file": rb.get("source_file")
    }

    return result


def finalize_result(
    result: Dict[str, Any],
    runbook_data: Optional[List[Dict[str, Any]]],
    attach_runbook: bool
) -> Dict[str, Any]:
    """
    Chuẩn hóa output cuối.
    """

    if not attach_runbook:
        return result

    runbooks = runbook_data or load_runbooks(RUNBOOK_JSON)
    runbook_index = build_runbook_index(runbooks)

    return attach_runbook_content(
        result=result,
        runbook_index=runbook_index
    )


# ========================
# MAIN RESOLVER
# ========================

def issue_resolver_V4_nho(
    email_text: str,
    resolved_service_id: str,
    keyword_entries: Optional[List[Dict[str, Any]]] = None,
    runbook_data: Optional[List[Dict[str, Any]]] = None,
    use_llm_fallback: bool = True,
    attach_runbook: bool = True
) -> Dict[str, Any]:
    """
    V4_nhỏ issue resolver.

    Input:
    - email_text: subject + body ServiceDesk email
    - resolved_service_id: output từ service_resolver_V4_nho

    Flow:
    1. Filter keyword_catalog theo service_id
    2. Build token stats trong service scope
    3. Field-priority search:
       - keyword_list
       - intents
       - patterns
       - synonyms
    4. Nếu chưa resolve: global token overlap
    5. Nếu vẫn fail/weak/ambiguous: LLM fallback Option 3
    6. Nếu có title: lookup runbook_data để trả steps
    """

    if not email_text or not str(email_text).strip():
        return {
            "resolved": False,
            "decision": "empty_email_text",
            "confidence": 0.0
        }

    if not resolved_service_id:
        return {
            "resolved": False,
            "decision": "missing_resolved_service_id",
            "confidence": 0.0
        }

    entries = keyword_entries or load_keyword_entries(KEYWORD_CATALOG_JSON)

    scoped_entries = filter_entries_by_service(
        entries=entries,
        resolved_service_id=resolved_service_id
    )

    if not scoped_entries:
        return {
            "resolved": False,
            "decision": "empty_issue_scope_for_service",
            "confidence": 0.0,
            "resolved_service_id": resolved_service_id
        }

    token_stats = build_scope_token_stats(scoped_entries)

    trace = {
        "resolved_service_id": resolved_service_id,
        "scope_size": len(scoped_entries),
        "token_stats": {
            "total_docs": token_stats.get("total_docs"),
            "doc_freq_size": len(token_stats.get("doc_freq", {}))
        }
    }

    # ========================
    # 1. Field-priority search
    # ========================

    priority_result = field_priority_search(
        email_text=email_text,
        scoped_entries=scoped_entries,
        token_stats=token_stats
    )

    if priority_result.get("resolved"):
        priority_result["resolver_flow"] = "field_priority"
        priority_result["trace"] = trace
        return finalize_result(priority_result, runbook_data, attach_runbook)

    # ========================
    # 2. Global token overlap
    # ========================

    token_result = global_token_overlap_search(
        email_text=email_text,
        scoped_entries=scoped_entries,
        token_stats=token_stats
    )

    if token_result.get("resolved"):
        token_result["resolver_flow"] = "global_token_overlap"
        token_result["previous_priority_decision"] = priority_result.get("decision")
        token_result["field_trace"] = priority_result.get("field_trace", [])
        token_result["trace"] = trace
        return finalize_result(token_result, runbook_data, attach_runbook)

    # ========================
    # 3. LLM fallback - Option 3
    # ========================

    if use_llm_fallback:
        llm_result = call_llm_fallback(
            email_text=email_text,
            resolved_service_id=resolved_service_id,
            scoped_entries=scoped_entries
        )

        llm_result["resolver_flow"] = "llm_fallback"
        llm_result["previous_priority_decision"] = priority_result.get("decision")
        llm_result["previous_token_decision"] = token_result.get("decision")
        llm_result["field_trace"] = priority_result.get("field_trace", [])
        llm_result["rule_top_k"] = token_result.get("top_k", [])
        llm_result["trace"] = trace

        return finalize_result(llm_result, runbook_data, attach_runbook)

    # ========================
    # 4. Deterministic failed, no LLM
    # ========================

    token_result["resolver_flow"] = "deterministic_failed_no_llm"
    token_result["previous_priority_decision"] = priority_result.get("decision")
    token_result["field_trace"] = priority_result.get("field_trace", [])
    token_result["trace"] = trace

    return token_result


# ========================
# QUICK TEST
# ========================

if __name__ == "__main__":

    test_email = """
    
    Subject: Outlook báo lỗi chứng chỉ

    Hi team,

    Người dùng phản ánh khi mở Outlook thì xuất hiện thông báo lỗi liên quan đến chứng chỉ bảo mật, nội dung đại khái là tên chứng chỉ không khớp với tên site.

    Hiện tại người dùng không thể truy cập mailbox bình thường, cần kiểm tra lại cấu hình dịch vụ Exchange.

    Nhờ hỗ trợ kiểm tra giúp.

    Thanks.

    """

    result = issue_resolver_V4_nho(
        email_text=test_email,
        resolved_service_id="6.13-Microsoft Team / Office 365",
        use_llm_fallback=False,
        attach_runbook=True
    )

    #print(json.dumps(result, ensure_ascii=False, indent=2))
    with open("debug_entry.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ Saved debug_entry.json")