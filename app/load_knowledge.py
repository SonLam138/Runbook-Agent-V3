import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

KNOWLEDGE_FILE = DATA_DIR / "knowledge_base.json"

# =====================================================
# LOAD ONCE
# =====================================================
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

# =====================================================
# PREBUILD CONTEXT
# =====================================================

# -------- General clarify context --------
GENERAL_CLARIFY_CONTEXT = "=== GENERAL IT SUPPORT CLARIFY RULES ===\n"
for r in KNOWLEDGE.get("general_rules", {}).get("clarify_rules", []):
    GENERAL_CLARIFY_CONTEXT += f"- {r}\n"

# -------- General decision context --------
GENERAL_DECISION_CONTEXT = "=== GENERAL IT SUPPORT DECISION RULES ===\n"
for r in KNOWLEDGE.get("general_rules", {}).get("decision_rules", []):
    GENERAL_DECISION_CONTEXT += f"- {r}\n"

# -------- Service specific cache --------
SERVICE_CLARIFY_CONTEXT = {}
SERVICE_DECISION_CONTEXT = {}

for item in KNOWLEDGE.get("services", []):
    service = item.get("service", "").strip()
    if not service:
        continue

    # Clarify context
    clarify_text = GENERAL_CLARIFY_CONTEXT
    clarify_text += f"\n=== SERVICE: {service} ===\n"

    if item.get("description"):
        clarify_text += f"Description: {item['description']}\n"

    clarify_text += "Clarify rules:\n"
    for r in item.get("clarify_rules", []):
        clarify_text += f"- {r}\n"

    SERVICE_CLARIFY_CONTEXT[service] = clarify_text

    # Decision context
    decision_text = GENERAL_DECISION_CONTEXT
    decision_text += f"\n=== SERVICE: {service} ===\n"

    if item.get("description"):
        decision_text += f"Description: {item['description']}\n"

    decision_text += "Decision rules:\n"
    for r in item.get("decision_rules", []):
        decision_text += f"- {r}\n"

    SERVICE_DECISION_CONTEXT[service] = decision_text


# =====================================================
# ACCESS FUNCTIONS
# =====================================================
def get_clarify_knowledge(service: str = None) -> str:
    """
    Return clarify knowledge context (prebuilt, no runtime rebuild).
    """
    if service and service in SERVICE_CLARIFY_CONTEXT:
        return SERVICE_CLARIFY_CONTEXT[service]
    return GENERAL_CLARIFY_CONTEXT


def get_decision_knowledge(service: str = None) -> str:
    """
    Return decision knowledge context (prebuilt, no runtime rebuild).
    """
    if service and service in SERVICE_DECISION_CONTEXT:
        return SERVICE_DECISION_CONTEXT[service]
    return GENERAL_DECISION_CONTEXT