import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_CATALOG_PATH = BASE_DIR / "knowledge_catalog_V4_nho.json"


# ============================================================
# GENERIC / WEAK TOKENS
# ============================================================

# Các token này không đủ để xác định service nếu đứng một mình
GENERIC_ACTION_TOKENS = {
    "kết", "nối",
    "đăng", "nhập",
    "login", "signin", "sign", "in",
    "gửi", "nhận",
    "mail", "email",
    "mật", "khẩu", "password",
    "truy", "cập", "access",
    "credential", "credentials", "rights",
    "ứng", "dụng", "app", "application",
    "lỗi", "error", "issue", "problem",
    "server",
    "mạng", "network",
    "nội", "bộ",
    "không", "được", "bị",
    "chậm", "treo", "timeout",
}

VI_STOPWORDS = {
    "tôi", "mình", "em", "anh", "chị",
    "không", "được", "bị", "lỗi",
    "vào", "ra", "trong", "ngoài",
    "sáng", "nay", "hôm", "qua",
    "cái", "này", "đó", "thì", "là",
    "với", "và", "hoặc", "nhưng",
    "khi", "có", "cho", "của",
    "xin", "nhờ", "giúp", "hỗ", "trợ",
    "kính", "gửi", "khối", "cntt",
    "chi", "tiết", "theo", "file", "đính", "kèm",
}

DOMAIN_CONFLICT_GROUPS = {
    "email": {
        "mail", "email", "outlook", "hộp", "thư"
    },
    "video_conference": {
        "hội", "nghị", "trực", "tuyến", "truyền", "hình", "họp"
    },
    "vpn": {
        "vpn", "remote", "mạng", "nội", "bộ"
    },
    "phone": {
        "phone", "ipphone", "điện", "thoại"
    }
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text:
    - lowercase
    - thay ký tự đặc biệt bằng space
    - giữ tiếng Việt có dấu
    - quan trọng: tách được các mã như 0.01-T24 -> 0 01 t24
    """
    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def raw_tokens(text: str) -> Set[str]:
    """
    Token không loại stopword.
    Dùng cho việc phân loại generic/action keyword.
    """
    text = normalize_text(text)
    return {t for t in text.split() if len(t) >= 2}


def tokenize(text: str) -> Set[str]:
    """
    Token phục vụ match nội dung.
    Có loại stopword.
    """
    tokens = raw_tokens(text)
    return {
        t for t in tokens
        if t not in VI_STOPWORDS
        and len(t) >= 2
        and not t.isdigit()
    }


def is_code_like_token(token: str) -> bool:
    """
    Token dạng mã service / technical code.
    Ví dụ:
    - t24
    - bpm
    - aws
    - crm
    - citad
    - swift
    - smartvista
    """
    if not token:
        return False

    token = token.lower().strip()

    if token.isdigit():
        return False

    # Có cả chữ và số: t24, teller7
    has_alpha = bool(re.search(r"[a-z]", token))
    has_digit = bool(re.search(r"\d", token))

    if has_alpha and has_digit:
        return True

    # Token tiếng Anh / mã ngắn thường là service code
    if re.fullmatch(r"[a-z][a-z0-9]{1,20}", token):
        # Không coi keyword generic là code
        if token not in GENERIC_ACTION_TOKENS and token not in VI_STOPWORDS:
            return True

    return False


def phrase_in_text(phrase: str, text: str) -> bool:
    """
    Exact phrase match trên normalized text.
    Dùng padding space để tránh match substring nguy hiểm.
    """
    phrase_norm = normalize_text(phrase)
    text_norm = normalize_text(text)

    if not phrase_norm or not text_norm:
        return False

    return f" {phrase_norm} " in f" {text_norm} "


def is_generic_pattern(pattern: str) -> bool:
    """
    Pattern được coi là generic nếu toàn bộ token thuộc nhóm generic/action/stopword.
    Ví dụ:
    - lỗi
    - truy cập
    - đăng nhập
    - mật khẩu
    """
    tokens = raw_tokens(pattern)

    if not tokens:
        return True

    return tokens.issubset(GENERIC_ACTION_TOKENS.union(VI_STOPWORDS))


# ============================================================
# EMAIL SECTION PARSING
# ============================================================

def parse_email_sections(email_text: str) -> Dict[str, str]:
    """
    Tách nhẹ final_email_text do email_processor build:

    Subject: ...
    Body:
    ...
    OCR_Text_From_Inline_Image:
    ...

    Nếu không có marker thì toàn bộ text đưa vào body.
    """
    if not email_text:
        return {
            "subject": "",
            "body": "",
            "ocr": "",
            "full": ""
        }

    subject = ""
    body = ""
    ocr = ""

    # Subject
    m_subject = re.search(
        r"Subject:\s*(.*?)(?:\n\s*Body:|\n\s*OCR_Text_From_Inline_Image:|$)",
        email_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m_subject:
        subject = m_subject.group(1).strip()

    # Body
    m_body = re.search(
        r"Body:\s*(.*?)(?:\n\s*OCR_Text_From_Inline_Image:|$)",
        email_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m_body:
        body = m_body.group(1).strip()

    # OCR
    m_ocr = re.search(
        r"OCR_Text_From_Inline_Image:\s*(.*)$",
        email_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m_ocr:
        ocr = m_ocr.group(1).strip()

    # Nếu không parse được marker thì fallback full vào body
    if not subject and not body and not ocr:
        body = email_text

    return {
        "subject": subject,
        "body": body,
        "ocr": ocr,
        "full": email_text
    }


# ============================================================
# CATALOG
# ============================================================

def load_knowledge_catalog() -> List[Dict[str, Any]]:
    with open(KNOWLEDGE_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "services" in data:
        return data["services"]

    return data


def build_service_profile(service: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build profile cho một service.

    Mục tiêu:
    - hard_anchor_tokens: mã / tên định danh mạnh
    - alias_phrases: phrase alias có thể exact match
    - descriptive_tokens: token mô tả service, yếu hơn anchor
    """

    service_id = service.get("service_id", "") or ""
    service_name = service.get("service_name", "") or ""
    aliases = service.get("aliases", []) or []
    description = service.get("description", "") or ""

    identity_texts = [service_id, service_name] + aliases

    hard_anchor_tokens: Set[str] = set()
    alias_phrases: List[str] = []
    descriptive_tokens: Set[str] = set()

    # 1. Tách token định danh mạnh từ service_id, service_name, aliases
    for text in identity_texts:
        tokens = raw_tokens(text)

        for t in tokens:
            if t in GENERIC_ACTION_TOKENS or t in VI_STOPWORDS:
                continue

            # Code-like token mạnh
            if is_code_like_token(t):
                hard_anchor_tokens.add(t)

        # Phrase alias/name dùng cho exact phrase
        text_norm = normalize_text(text)
        if text_norm and not is_generic_pattern(text_norm):
            alias_phrases.append(text)

    # 2. Descriptive tokens từ service_name + description
    desc_source = f"{service_name} {description}"
    for t in tokenize(desc_source):
        if t not in hard_anchor_tokens and t not in GENERIC_ACTION_TOKENS:
            descriptive_tokens.add(t)

    return {
        "hard_anchor_tokens": hard_anchor_tokens,
        "alias_phrases": alias_phrases,
        "descriptive_tokens": descriptive_tokens,
    }


# ============================================================
# CONFLICT
# ============================================================

def detect_cross_domain_conflict(clean_text: str) -> Dict[str, Any]:
    tokens = tokenize(clean_text)
    matched_domains = {}

    for domain, domain_tokens in DOMAIN_CONFLICT_GROUPS.items():
        overlap = tokens.intersection(domain_tokens)
        if overlap:
            matched_domains[domain] = sorted(list(overlap))

    return {
        "has_conflict": len(matched_domains) >= 2,
        "matched_domains": matched_domains
    }


# ============================================================
# SCORING
# ============================================================

FIELD_WEIGHTS = {
    "subject": 1.6,
    "body": 1.0,
    "ocr": 0.8,
}

EVIDENCE_RANK = {
    "none": 0,
    "weak_description": 1,
    "anchor_plus_context": 2,
    "alias_phrase": 3,
    "strong_anchor": 4,
}


def score_service_against_email(
    service: Dict[str, Any],
    sections: Dict[str, str]
) -> Dict[str, Any]:
    """
    Tính score service theo evidence-based scoring.

    Rule chính:
    - Generic/action keyword không được tạo candidate.
    - Issue keyword chỉ support nếu đã có anchor.
    - Hard anchor thắng tuyệt đối generic.
    """

    service_id = service.get("service_id")
    service_name = service.get("service_name", "")

    aliases = service.get("aliases", []) or []
    symptoms = service.get("symptoms", []) or []
    common_keywords = service.get("common_keywords", []) or []
    description = service.get("description", "") or ""

    profile = build_service_profile(service)

    hard_anchor_tokens = profile["hard_anchor_tokens"]
    alias_phrases = profile["alias_phrases"]
    descriptive_tokens = profile["descriptive_tokens"]

    score = 0.0
    evidence_level = "none"
    evidence_trace: List[Dict[str, Any]] = []

    matched = {
        "hard_anchors": [],
        "alias_phrases": [],
        "description": [],
        "support_keywords": [],
        "support_symptoms": [],
        "ignored_generic": [],
    }

    # --------------------------------------------------------
    # 1. Strong anchor token match
    # --------------------------------------------------------
    anchor_found = False

    for field_name in ["subject", "body", "ocr"]:
        field_text = sections.get(field_name, "") or ""
        field_tokens = raw_tokens(field_text)
        field_weight = FIELD_WEIGHTS[field_name]

        overlap = field_tokens.intersection(hard_anchor_tokens)

        for token in sorted(overlap):
            anchor_found = True
            add_score = 10.0 * field_weight
            score += add_score

            matched["hard_anchors"].append({
                "token": token,
                "field": field_name,
                "score": round(add_score, 3),
            })

            evidence_trace.append({
                "type": "strong_anchor",
                "token": token,
                "field": field_name,
                "score": round(add_score, 3),
            })

    if anchor_found:
        evidence_level = "strong_anchor"

    # --------------------------------------------------------
    # 2. Exact alias phrase match
    # --------------------------------------------------------
    for phrase in alias_phrases:
        if is_generic_pattern(phrase):
            continue

        for field_name in ["subject", "body", "ocr"]:
            field_text = sections.get(field_name, "") or ""
            field_weight = FIELD_WEIGHTS[field_name]

            if phrase_in_text(phrase, field_text):
                add_score = 7.0 * field_weight
                score += add_score

                matched["alias_phrases"].append({
                    "phrase": phrase,
                    "field": field_name,
                    "score": round(add_score, 3),
                })

                evidence_trace.append({
                    "type": "alias_phrase",
                    "phrase": phrase,
                    "field": field_name,
                    "score": round(add_score, 3),
                })

                if EVIDENCE_RANK[evidence_level] < EVIDENCE_RANK["alias_phrase"]:
                    evidence_level = "alias_phrase"

    # --------------------------------------------------------
    # 3. Support keywords: only if anchor exists
    # --------------------------------------------------------
    for keyword in common_keywords:
        keyword_norm = normalize_text(keyword)

        if not keyword_norm:
            continue

        if is_generic_pattern(keyword):
            matched["ignored_generic"].append({
                "pattern": keyword,
                "reason": "generic_keyword_not_allowed_to_create_service_candidate"
            })
            continue

        for field_name in ["subject", "body", "ocr"]:
            field_text = sections.get(field_name, "") or ""
            field_weight = FIELD_WEIGHTS[field_name]

            if phrase_in_text(keyword, field_text):
                if anchor_found:
                    add_score = 1.5 * field_weight
                    score += add_score

                    matched["support_keywords"].append({
                        "keyword": keyword,
                        "field": field_name,
                        "score": round(add_score, 3),
                    })

                    evidence_trace.append({
                        "type": "support_keyword",
                        "keyword": keyword,
                        "field": field_name,
                        "score": round(add_score, 3),
                    })

                    if EVIDENCE_RANK[evidence_level] < EVIDENCE_RANK["anchor_plus_context"]:
                        evidence_level = "anchor_plus_context"
                else:
                    matched["ignored_generic"].append({
                        "pattern": keyword,
                        "reason": "keyword_without_service_anchor"
                    })

    # --------------------------------------------------------
    # 4. Support symptoms: only if anchor exists
    # --------------------------------------------------------
    for symptom in symptoms:
        symptom_norm = normalize_text(symptom)

        if not symptom_norm:
            continue

        for field_name in ["subject", "body", "ocr"]:
            field_text = sections.get(field_name, "") or ""
            field_weight = FIELD_WEIGHTS[field_name]

            if phrase_in_text(symptom, field_text):
                if anchor_found:
                    add_score = 1.2 * field_weight
                    score += add_score

                    matched["support_symptoms"].append({
                        "symptom": symptom,
                        "field": field_name,
                        "score": round(add_score, 3),
                    })

                    evidence_trace.append({
                        "type": "support_symptom",
                        "symptom": symptom,
                        "field": field_name,
                        "score": round(add_score, 3),
                    })

                    if EVIDENCE_RANK[evidence_level] < EVIDENCE_RANK["anchor_plus_context"]:
                        evidence_level = "anchor_plus_context"
                else:
                    matched["ignored_generic"].append({
                        "pattern": symptom,
                        "reason": "symptom_without_service_anchor"
                    })

    # --------------------------------------------------------
    # 5. Weak description overlap
    # --------------------------------------------------------
    # Chỉ dùng khi chưa có anchor/alias mạnh.
    # Không để description thắng hard anchor.
    if evidence_level == "none" and descriptive_tokens:
        full_tokens = tokenize(sections.get("full", ""))
        overlap = full_tokens.intersection(descriptive_tokens)

        coverage = len(overlap) / max(len(descriptive_tokens), 1)

        if len(overlap) >= 2 and coverage >= 0.45:
            add_score = len(overlap) * coverage * 2.0
            score += add_score

            matched["description"].append({
                "overlap_tokens": sorted(list(overlap)),
                "coverage": round(coverage, 3),
                "score": round(add_score, 3),
            })

            evidence_trace.append({
                "type": "weak_description",
                "overlap_tokens": sorted(list(overlap)),
                "coverage": round(coverage, 3),
                "score": round(add_score, 3),
            })

            evidence_level = "weak_description"

    # --------------------------------------------------------
    # Candidate rule
    # --------------------------------------------------------
    # Chỉ những service có evidence thật mới được candidate.
    is_candidate = evidence_level != "none"

    return {
        "service_id": service_id,
        "service_name": service_name,
        "score": round(score, 3),
        "evidence_level": evidence_level,
        "is_candidate": is_candidate,
        "matched": matched,
        "evidence_trace": evidence_trace,
    }


def compute_confidence(
    best: Dict[str, Any],
    second: Optional[Dict[str, Any]],
) -> float:
    """
    Confidence dựa trên:
    - quality của evidence
    - score
    - margin với candidate thứ 2
    """

    evidence_level = best.get("evidence_level", "none")
    score = best.get("score", 0.0)

    base_by_level = {
        "strong_anchor": 0.9,
        "alias_phrase": 0.82,
        "anchor_plus_context": 0.74,
        "weak_description": 0.55,
        "none": 0.0,
    }

    confidence = base_by_level.get(evidence_level, 0.0)

    # Score bonus nhẹ
    if score >= 12:
        confidence += 0.05
    elif score >= 8:
        confidence += 0.03

    # Margin bonus/penalty
    if second and second.get("score", 0) > 0:
        gap = score - second["score"]
        margin = gap / max(score, 1.0)

        if margin >= 0.5:
            confidence += 0.03
        elif margin < 0.2:
            confidence -= 0.08

    confidence = max(0.0, min(1.0, confidence))
    return round(confidence, 3)


def is_ambiguous(
    best: Dict[str, Any],
    second: Optional[Dict[str, Any]],
    ambiguity_margin: float
) -> bool:
    if not second:
        return False

    if second.get("score", 0) <= 0:
        return False

    best_level = best.get("evidence_level", "none")
    second_level = second.get("evidence_level", "none")

    best_score = best.get("score", 0.0)
    second_score = second.get("score", 0.0)

    # Nếu cả hai đều có strong anchor thì ambiguous thật sự
    if best_level == "strong_anchor" and second_level == "strong_anchor":
        gap = best_score - second_score
        margin = gap / max(best_score, 1.0)
        return margin < 0.35

    # Ambiguity score thông thường
    gap = best_score - second_score
    margin = gap / max(best_score, 1.0)

    return margin < ambiguity_margin


# ============================================================
# MAIN RESOLVER
# ============================================================

def service_resolver_v4_nho(
    clean_text: str,
    knowledge_catalog: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3,
    min_confidence: float = 0.60,
    ambiguity_margin: float = 0.20,
) -> Dict[str, Any]:
    """
    Evidence-based service resolver cho V4_nhỏ.

    Naming note:
    - Logic này là v3 về kiến trúc.
    - Nhưng function name giữ là service_resolver_v4_nho để tránh đổi project.

    Priority:
    1. Hard service anchor/code
    2. Alias phrase
    3. Anchor + support context
    4. Description overlap
    5. Generic/action keyword: ignored
    6. BoW: support-only
    7. Ambiguous multi-service: fallback
    """

    if knowledge_catalog is None:
        knowledge_catalog = load_knowledge_catalog()

    sections = parse_email_sections(clean_text)
    normalized_full_text = normalize_text(sections.get("full", ""))

    scored_services = []

    for service in knowledge_catalog:
        result = score_service_against_email(service, sections)

        if result["is_candidate"]:
            scored_services.append(result)

    scored_services = sorted(
        scored_services,
        key=lambda x: (
            EVIDENCE_RANK.get(x.get("evidence_level", "none"), 0),
            x.get("score", 0.0)
        ),
        reverse=True
    )

    top_candidates = scored_services[:top_k]
    conflict_result = detect_cross_domain_conflict(normalized_full_text)

    # --------------------------------------------------------
    # No valid candidate
    # --------------------------------------------------------
    if not top_candidates:
        return {
            "resolved": False,
            "service_id": None,
            "service_name": None,
            "confidence": 0.0,
            "decision": "no_service_signal",
            "need_llm_fallback": True,
            "evidence_level": "none",
            "evidence": [],
            "conflict": conflict_result,
            "top_k": [],
        }

    best = top_candidates[0]
    second = top_candidates[1] if len(top_candidates) > 1 else None

    confidence = compute_confidence(best, second)
    ambiguous = is_ambiguous(best, second, ambiguity_margin)

    decision = "resolved_by_rule"
    need_llm_fallback = False

    # --------------------------------------------------------
    # Decision rules
    # --------------------------------------------------------
    if best["evidence_level"] == "strong_anchor":
        decision = "resolved_by_strong_anchor"

    elif best["evidence_level"] == "alias_phrase":
        decision = "resolved_by_alias_phrase"

    elif best["evidence_level"] == "anchor_plus_context":
        decision = "resolved_by_anchor_plus_context"

    elif best["evidence_level"] == "weak_description":
        decision = "resolved_by_weak_description"

    else:
        decision = "no_service_signal"
        need_llm_fallback = True

    if confidence < min_confidence:
        need_llm_fallback = True
        decision = "low_confidence"

    if ambiguous:
        need_llm_fallback = True
        decision = "ambiguous_top_candidates"

    if conflict_result["has_conflict"]:
        need_llm_fallback = True
        decision = "cross_domain_conflict"

    # --------------------------------------------------------
    # Important safety:
    # Nếu fallback thì không trả service_id như resolved thật.
    # Tránh downstream dùng nhầm service sai.
    # --------------------------------------------------------
    if need_llm_fallback:
        return {
            "resolved": False,
            "service_id": None,
            "service_name": None,
            "confidence": confidence,
            "decision": decision,
            "need_llm_fallback": True,
            "evidence_level": best.get("evidence_level"),
            "evidence": best.get("evidence_trace", []),
            "candidate_service": {
                "service_id": best.get("service_id"),
                "service_name": best.get("service_name"),
                "score": best.get("score"),
                "evidence_level": best.get("evidence_level"),
                "matched": best.get("matched"),
            },
            "conflict": conflict_result,
            "top_k": top_candidates,
        }

    return {
        "resolved": True,
        "service_id": best["service_id"],
        "service_name": best["service_name"],
        "confidence": confidence,
        "decision": decision,
        "need_llm_fallback": False,
        "evidence_level": best.get("evidence_level"),
        "evidence": best.get("evidence_trace", []),
        "conflict": conflict_result,
        "top_k": top_candidates,
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================
# Để run_v4_nho cũ vẫn dùng được:
# from service_resolver_V4_nho import service_resolver_v2

def service_resolver_v2(
    clean_text: str,
    knowledge_catalog: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3,
    min_confidence: float = 0.60,
    ambiguity_margin: float = 0.20,
) -> Dict[str, Any]:
    return service_resolver_v4_nho(
        clean_text=clean_text,
        knowledge_catalog=knowledge_catalog,
        top_k=top_k,
        min_confidence=min_confidence,
        ambiguity_margin=ambiguity_margin,
    )