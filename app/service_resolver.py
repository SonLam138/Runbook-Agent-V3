import json
import re
from pathlib import Path
import numpy as np
import faiss

from app.load_model import embedding_model

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SERVICE_CATALOG_FILE = DATA_DIR / "service_catalog.json"

SERVICE_SCORE_THRESHOLD = 0.55
SERVICE_STRONG_THRESHOLD = 0.72

# =====================================================
# LOAD CATALOG
# =====================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SERVICE_CATALOG = load_json(SERVICE_CATALOG_FILE)

# mỗi service -> 1 text để embed
SERVICE_TEXTS = []
SERVICE_ITEMS = []

for item in SERVICE_CATALOG:
    service = item.get("service", "")
    keywords = ", ".join(item.get("keywords", []))
    aliases = ", ".join(item.get("aliases", []))
    desc = item.get("description", "")

    text = f"""
Service: {service}
Aliases: {aliases}
Keywords: {keywords}
Description: {desc}
""".strip()

    SERVICE_TEXTS.append(text)
    SERVICE_ITEMS.append(item)

SERVICE_EMBEDDINGS = embedding_model.encode(
    SERVICE_TEXTS,
    normalize_embeddings=True
)
SERVICE_EMBEDDINGS = np.array(SERVICE_EMBEDDINGS, dtype="float32")

SERVICE_INDEX = faiss.IndexFlatIP(SERVICE_EMBEDDINGS.shape[1])
SERVICE_INDEX.add(SERVICE_EMBEDDINGS)

# =====================================================
# HELPERS
# =====================================================
def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def rule_match_service(query: str):
    """
    Match nhanh bằng keyword / aliases trong service catalog.
    Return:
        (service_name, score, source)
    """
    q = normalize(query)

    best_service = None
    best_score = 0.0

    for item in SERVICE_ITEMS:
        service = item.get("service", "")
        aliases = item.get("aliases", [])
        keywords = item.get("keywords", [])

        # exact / contains kiểu deterministic
        score = 0.0

        all_terms = [service] + aliases + keywords
        for term in all_terms:
            t = normalize(term)
            if not t:
                continue

            if q == t:
                score = max(score, 1.0)
            elif t in q:
                score = max(score, 0.9)
            elif q in t and len(q) >= 3:
                score = max(score, 0.8)

        if score > best_score:
            best_score = score
            best_service = service

    if best_service:
        return {
            "service": best_service,
            "score": best_score,
            "source": "rule"
        }

    return None


def semantic_match_service(query: str):
    """
    Semantic match service bằng embedding + FAISS
    """
    q_vec = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )
    q_vec = np.array(q_vec, dtype="float32")

    scores, indices = SERVICE_INDEX.search(q_vec, 1)

    score = float(scores[0][0])
    idx = int(indices[0][0])

    if idx == -1:
        return None

    item = SERVICE_ITEMS[idx]

    if score < SERVICE_SCORE_THRESHOLD:
        return None

    return {
        "service": item.get("service", ""),
        "score": score,
        "source": "semantic"
    }


def resolve_service(query: str, state=None):
    """
    V1:
    1. Rule-based match
    2. Semantic match
    3. Không dùng LLM

    Return:
    {
        "service": "...",
        "score": 0.88,
        "source": "rule|semantic",
        "is_strong": True/False
    }
    hoặc None
    """
    # 1) rule first
    rule_hit = rule_match_service(query)
    if rule_hit:
        rule_hit["is_strong"] = rule_hit["score"] >= 0.85
        return rule_hit

    # 2) semantic fallback
    sem_hit = semantic_match_service(query)
    if sem_hit:
        sem_hit["is_strong"] = sem_hit["score"] >= SERVICE_STRONG_THRESHOLD
        return sem_hit

    return None