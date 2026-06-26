import pandas as pd
import json
import re
import unicodedata


# =========================
# NORMALIZE FUNCTIONS
# =========================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower().strip()

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text


def remove_vietnamese_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFD", text)
    text = "".join([c for c in text if unicodedata.category(c) != "Mn"])
    text = text.replace("đ", "d")
    return text


def normalize_advanced(text: str) -> str:
    text = normalize_text(text)
    text = remove_vietnamese_accents(text)
    return text


def parse_list_field(text: str):
    """
    Split by comma, semicolon, newline
    Normalize + deduplicate
    Preserve domain/URL
    Enrich domain variants (http <-> domain)
    """

    if not isinstance(text, str) or not text.strip():
        return []

    import re

    # ✅ split đúng (fix bug cũ)
    items = re.split(r"[,;\n]", text)

    cleaned = []

    for item in items:
        item = item.strip()

        if not item:
            continue

        # normalize cơ bản (KHÔNG phá domain)
        normalized = normalize_text(item)

        if normalized:
            cleaned.append(normalized)

    # ✅ dedupe lần 1
    cleaned = set(cleaned)

    # =========================
    # ✅ ENRICH DOMAIN / URL
    # =========================
    enriched = set()

    for item in cleaned:
        enriched.add(item)

        # ✅ nếu là URL → add domain
        if item.startswith("http://") or item.startswith("https://"):
            domain = item.replace("http://", "").replace("https://", "")
            domain = domain.strip("/")

            if domain:
                enriched.add(domain)

        # ✅ nếu là domain → add https version
        elif "." in item and not item.startswith("http"):
            enriched.add(f"https://{item}")

    # =========================
    # ✅ FINAL CLEAN
    # =========================
    final = []

    for item in enriched:
        item = item.strip()

        if item:
            final.append(item)

    # ✅ sort để ổn định output (debug rất dễ)
    return sorted(set(final))



# =========================
# SERVICE ID BUILDER
# =========================

def generate_service_id(service_name: str):
    if not service_name:
        return ""

    name = normalize_advanced(service_name)

    # replace space -> _
    name = name.replace(" ", "_")

    return name


# =========================
# MAIN BUILD FUNCTION
# =========================

def build_catalog(excel_path, output_path):
    df = pd.read_excel(excel_path)

    catalog = []

    for idx, row in df.iterrows():
        service_name = str(row.get("ServiceName", "")).strip()
        description = str(row.get("Description", "")).strip()

        alias_raw = row.get("Alias", "")
        symptom_raw = row.get("Symptom", "")
        keyword_raw = row.get("Keyword", "")

        # parse fields
        aliases = parse_list_field(alias_raw)
        symptoms = parse_list_field(symptom_raw)
        keywords = parse_list_field(keyword_raw)

        # generate service_id
        service_id = generate_service_id(service_name)

        # final object
        service_obj = {
            "service_id": service_id,
            "service_name": service_name,
            "description": description,
            "aliases": aliases,
            "symptoms": symptoms,
            "common_keywords": keywords
        }

        catalog.append(service_obj)

    # write file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"✅ Build xong catalog: {output_path}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    build_catalog(
        excel_path=f"D:\ServiceCatalog\P_service.xlsx",
        output_path=f"D:\Runbook_Agent\V4_nho\knowledge_cata_test.json"
    )