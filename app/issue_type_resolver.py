import json
import re
from pathlib import Path


# =========================
# LOAD KEYWORD CATALOG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KEYWORD_FILE = DATA_DIR / "keyword_catalog.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def as_list(value):
    """
    Helper:
    - Nếu value là list -> giữ nguyên
    - Nếu value là str -> tách theo dấu phẩy nếu có
    - Nếu None -> []
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]

    return []


def build_keyword_catalog(raw_catalog):
    """
    Chuẩn hóa catalog thô từ RB_build_All_in_1 về dạng:

    {
        "Service A": [
            {...},
            {...}
        ],
        "Service B": [
            {...}
        ]
    }

    Lý do cần hàm này:
    - keyword_catalog.json có thể là list hoặc dict
    - data được sinh tự động từ pipeline, không sửa tay file JSON
    - rule engine cần lookup nhanh theo service

    TRAINING NOTE:
    Đây là normalization layer giữa raw pipeline data và rule engine.
    Nó giúp log/training data ổn định hơn dù format JSON sinh ra thay đổi.
    """

    # Case 1: Catalog đã là dict service -> items
    if isinstance(raw_catalog, dict):
        normalized_catalog = {}

        for service, items in raw_catalog.items():
            if not service:
                continue

            if isinstance(items, list):
                normalized_catalog[service] = items
            else:
                normalized_catalog[service] = [items]

        return normalized_catalog

    # Case 2: Catalog là list các item
    if isinstance(raw_catalog, list):
        normalized_catalog = {}

        for item in raw_catalog:
            # Nếu item ở top-level là string thì không biết thuộc service nào
            # nên bỏ qua để tránh match sai service.
            if isinstance(item, str):
                continue

            if not isinstance(item, dict):
                continue

            service = item.get("service")
            if not service:
                continue

            if service not in normalized_catalog:
                normalized_catalog[service] = []

            normalized_catalog[service].append(item)

        return normalized_catalog

    # Fallback
    return {}


RAW_KEYWORD_CATALOG = load_json(KEYWORD_FILE)
KEYWORD_CATALOG = build_keyword_catalog(RAW_KEYWORD_CATALOG)


# =========================
# HELPERS
# =========================

def get_keywords_for_service(service):
    """
    Lấy keyword list theo service.

    Có hỗ trợ fallback case-insensitive để tránh lỗi do service khác hoa/thường.
    """

    if not service:
        return []

    # Exact match trước
    if service in KEYWORD_CATALOG:
        return KEYWORD_CATALOG.get(service, [])

    # Case-insensitive fallback
    service_norm = normalize(service)

    for catalog_service, items in KEYWORD_CATALOG.items():
        if normalize(catalog_service) == service_norm:
            return items

    return []


def normalize_keyword_items(items):
    """
    Chuẩn hóa từng keyword item về dạng thống nhất:

    [
        {
            "issue_type": "...",
            "patterns": [...]
        }
    ]

    Input có thể gồm:
    - str
    - dict có patterns
    - dict có intents
    - dict có keyword
    - dict có title / issue_type

    TRAINING NOTE:
    Output chuẩn hóa này rất quan trọng nếu sau này dùng log để train:
    query -> matched_pattern -> issue_type
    """

    normalized_items = []

    for item in items:
        # Case 1: item là string
        if isinstance(item, str):
            value = item.strip()
            if not value:
                continue

            normalized_items.append({
                "issue_type": value,
                "patterns": [value]
            })
            continue

        # Case 2: item là dict
        if isinstance(item, dict):
            issue_type = (
                item.get("issue_type")
                or item.get("keyword")
                or item.get("title")
                or item.get("runbook_title")
            )

            patterns = []

            # Ưu tiên patterns nếu có
            patterns.extend(as_list(item.get("patterns")))

            # Nếu có intents thì cũng dùng làm patterns
            patterns.extend(as_list(item.get("intents")))

            # Nếu keyword là chuỗi nhiều từ khóa, dùng luôn làm patterns
            patterns.extend(as_list(item.get("keyword")))

            # Nếu title có giá trị, dùng thêm title làm pattern phụ
            patterns.extend(as_list(item.get("title")))

            # Nếu không có issue_type nhưng có pattern đầu tiên thì lấy pattern đầu làm issue_type
            if not issue_type and patterns:
                issue_type = patterns[0]

            # Nếu vẫn không có gì thì bỏ qua
            if not issue_type and not patterns:
                continue

            # Loại pattern rỗng + duplicate sau normalize
            clean_patterns = []
            seen = set()

            for p in patterns:
                p_norm = normalize(p)
                if not p_norm:
                    continue

                if p_norm in seen:
                    continue

                seen.add(p_norm)
                clean_patterns.append(p)

            # Nếu không có pattern nào thì dùng issue_type làm pattern fallback
            if not clean_patterns and issue_type:
                clean_patterns = [issue_type]

            normalized_items.append({
                "issue_type": issue_type,
                "patterns": clean_patterns
            })

    return normalized_items


# =========================
# RULE MATCH (CORE)
# =========================

def rule_match_issue_type(query: str, service: str):
    """
    Rule-based match:
    - chỉ match trong keyword của service đã xác định
    - return None nếu không match

    MONITORING NOTE:
    Nếu function này return None nhiều, nghĩa là keyword catalog chưa đủ tốt
    hoặc user query quá mơ hồ.

    TRAINING NOTE:
    matched_text và issue_type là signal rất quan trọng để sau này train
    classifier/intent resolver cho issue_type.
    """

    if not service:
        return None

    query_norm = normalize(query)

    raw_items = get_keywords_for_service(service)
    keyword_items = normalize_keyword_items(raw_items)

    for item in keyword_items:
        issue_type = item.get("issue_type")
        patterns = item.get("patterns", [])

        for p in patterns:
            p_norm = normalize(p)

            if not p_norm:
                continue

            if p_norm in query_norm:
                return {
                    "issue_type": issue_type,
                    "source": "rule",
                    "confidence": 1.0,
                    "matched_text": p,
                    "needs_more": False,
                    "reason": "matched_rule"
                }

    return None


# =========================
# MAIN RESOLVER
# =========================

def resolve_issue_type(query: str, service: str, state=None):
    """
    Flow:
    1. rule match
    2. nếu fail → trả needs_more

    TRAINING NOTE:
    Nếu no_match, đây là dữ liệu ambiguous/negative sample tốt
    để sau này cải thiện rule engine hoặc train issue_type classifier.
    """

    # ===== RULE MATCH =====
    rule_result = rule_match_issue_type(query, service)
    if rule_result:
        return rule_result

    # ===== FALLBACK CHƯA LÀM (LLM sẽ thêm sau) =====
    return {
        "issue_type": None,
        "source": "unknown",
        "confidence": 0.0,
        "matched_text": None,
        "needs_more": True,
        "reason": "no_match"
    }