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


KEYWORD_CATALOG = load_json(KEYWORD_FILE)


# =========================
# HELPERS
# =========================

def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# RULE MATCH (CORE)
# =========================

def rule_match_issue_type(query: str, service: str):
    """
    Rule-based match:
    - chỉ match trong keyword của service đã xác định
    - return None nếu không match
    """
    if not service:
        return None

    query_norm = normalize(query)
    keywords = KEYWORD_CATALOG.get(service, [])

    for item in keywords:
        patterns = item.get("patterns", [])
        for p in patterns:
            if normalize(p) in query_norm:
                return {
                    "issue_type": item.get("keyword"),
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
    2. nếu fail → trả needs_more (chưa làm LLM ở bước này)
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