import json
import hashlib
from pathlib import Path

from service_resolver_V4_nho import normalize_text

#import sys
#import io

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ========================
# PATH CONFIG
# ========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent   # đúng với project D:\Runbook_Agent

DATA_DIR = PROJECT_ROOT / "V4_nho"
SOURCE_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUNBOOK_JSON = SOURCE_DIR / "runbook_data.json"
MAPPING_JSON = DATA_DIR / "service_mapping.json"

OUTPUT_JSON = DATA_DIR / "keyword_catalog_V4_nho.json"
REPORT_JSON = DATA_DIR / "keyword_catalog_build_report_V4_nho.json"


# ========================
# BASIC UTILS
# ========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dedupe_keep_order(items):
    """
    Deduplicate nhưng giữ nguyên thứ tự xuất hiện.
    Dùng normalize_text để tránh duplicate do khác hoa/thường/khoảng trắng.
    """
    seen = set()
    result = []

    for item in items:
        if item is None:
            continue

        item = str(item).strip()

        if not item:
            continue

        key = normalize_text(item)

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def split_keyword_string(keyword):
    """
    keyword_list của V4_nhỏ được build từ field keyword trong runbook_data.

    Ví dụ:
    "Retention Policy, Mailbox đầy dung lượng, xóa hoặc archive mailbox định kỳ"

    ->
    [
      "Retention Policy",
      "Mailbox đầy dung lượng",
      "xóa hoặc archive mailbox định kỳ"
    ]

    Chú ý:
    - Không lấy runbook_data.keyword_list nữa.
    - Vì keyword là field cần monitor/enrich upstream.
    """
    if not keyword:
        return []

    text = str(keyword)

    # Chuẩn hóa một số delimiter phổ biến về dấu phẩy
    text = text.replace(";", ",")
    text = text.replace("|", ",")
    text = text.replace("\n", ",")

    parts = []

    for chunk in text.split(","):
        chunk = chunk.strip()

        if chunk:
            parts.append(chunk)

    return dedupe_keep_order(parts)


def normalize_list_field(value):
    """
    Đảm bảo field dạng list luôn convert an toàn.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return dedupe_keep_order(value)

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [value]

    return [str(value).strip()]


def stable_hash(obj):
    """
    Hash ổn định cho entry.
    Dùng để detect thay đổi source sau này.
    """
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ========================
# SERVICE MAPPING INDEX
# ========================

def build_mapping_index(service_mapping):
    """
    service_mapping.json format:

    {
      "6.01-Thư điện tử": {
        "service_name": "Thư điện tử",
        "description": "...",
        "technical_services": [
          "Exchange 2016",
          "Microsoft Exchange"
        ]
      }
    }

    Output:
    normalized technical service -> list business mapping records

    Lý do dùng list:
    - phòng trường hợp 1 technical service map vào nhiều business service.
    - hiện tại nếu clean tốt thì thường chỉ có 1.
    """

    index = {}

    for service_id, item in service_mapping.items():
        service_name = item.get("service_name", "")
        technical_services = item.get("technical_services", [])

        candidates = []

        # Cho phép match cả service_name
        if service_name:
            candidates.append(service_name)

        # Match trên technical_services là chính
        candidates.extend(technical_services)

        for name in candidates:
            norm = normalize_text(name)

            if not norm:
                continue

            index.setdefault(norm, []).append({
                "service_id": service_id,
                "service_name": service_name,
                "technical_services": technical_services
            })

    return index


def resolve_business_services_for_runbook(rb_service, mapping_index):
    """
    Map runbook.service -> service_id thông qua service_mapping.

    Chỉ dùng exact normalized match để tránh mapping ẩu.
    Không dùng fuzzy ở bước này.
    """

    norm = normalize_text(rb_service)

    if not norm:
        return []

    return mapping_index.get(norm, [])


# ========================
# ENTRY BUILDER
# ========================

def build_keyword_entry(runbook, business_mapping):
    """
    Mỗi entry = 1 runbook searchable index.

    Source of truth:
    - runbook_data.json

    Mapping/service boundary:
    - service_mapping.json

    Không lấy:
    - precheck
    - postcheck
    - steps
    - knowledge_catalog
    - description_short
    """

    title = str(runbook.get("title", "")).strip()
    service = str(runbook.get("service", "")).strip()

    # 1. keyword giữ raw từ runbook_data.keyword
    keyword_raw = str(runbook.get("keyword", "")).strip()

    # 2. keyword_list chỉ split từ keyword_raw
    keyword_list = split_keyword_string(keyword_raw)

    # 3. intents lấy từ description
    # Trong file gốc, description chính là kịch bản sử dụng.
    description = str(runbook.get("description", "")).strip()
    intents = dedupe_keep_order([description]) if description else []

    # 4. patterns lấy từ runbook_data.intents
    # Không lấy precheck/postcheck/steps/knowledge_catalog.
    patterns = normalize_list_field(runbook.get("intents", []))

    # 5. synonyms phase này để trống
    synonyms = []

    hash_payload = {
        "title": title,
        "service": service,
        "service_id": business_mapping.get("service_id"),
        "keyword": keyword_raw,
        "keyword_list": keyword_list,
        "intents": intents,
        "patterns": patterns,
        "source_file": runbook.get("source_file", "")
    }

    entry = {
        "title": title,
        "service": service,

        "service_id": business_mapping.get("service_id"),
        "service_name": business_mapping.get("service_name"),
        "technical_services": business_mapping.get("technical_services", []),

        "keyword": keyword_raw,
        "keyword_list": keyword_list,

        "intents": intents,
        "patterns": patterns,
        "synonyms": synonyms,

        "priority": 1.0,

        "source_file": runbook.get("source_file", ""),
        "source_hash": stable_hash(hash_payload)
    }

    return entry


def has_search_signal(entry):
    """
    Entry phải có ít nhất một vùng search có nghĩa.
    Nếu không có keyword_list/intents/patterns/synonyms thì resolver không dùng được.
    """
    return bool(
        entry.get("keyword_list")
        or entry.get("intents")
        or entry.get("patterns")
        or entry.get("synonyms")
    )


# ========================
# MAIN BUILD LOGIC
# ========================

def build_keyword_catalog():
    print("RUNBOOK_JSON:", RUNBOOK_JSON)
    print("MAPPING_JSON:", MAPPING_JSON)
    print("OUTPUT_JSON:", OUTPUT_JSON)

    runbook_data = load_json(RUNBOOK_JSON)
    service_mapping = load_json(MAPPING_JSON)

    mapping_index = build_mapping_index(service_mapping)

    entries = []
    unmatched_runbooks = []
    empty_signal_runbooks = []
    multi_mapping_runbooks = []

    for rb in runbook_data:
        title = str(rb.get("title", "")).strip()
        rb_service = str(rb.get("service", "")).strip()

        if not title:
            continue

        matched_business_services = resolve_business_services_for_runbook(
            rb_service=rb_service,
            mapping_index=mapping_index
        )

        if not matched_business_services:
            unmatched_runbooks.append({
                "title": title,
                "service": rb_service,
                "reason": "runbook_service_not_found_in_service_mapping"
            })
            continue

        if len(matched_business_services) > 1:
            multi_mapping_runbooks.append({
                "title": title,
                "service": rb_service,
                "matched_service_ids": [
                    m.get("service_id") for m in matched_business_services
                ],
                "reason": "runbook_service_matched_multiple_business_services"
            })

        for business_mapping in matched_business_services:
            service_id = business_mapping.get("service_id")

            entry = build_keyword_entry(
                runbook=rb,
                business_mapping=business_mapping
            )

            if not has_search_signal(entry):
                empty_signal_runbooks.append({
                    "title": title,
                    "service": rb_service,
                    "service_id": service_id,
                    "reason": "no_keyword_intent_pattern_synonym"
                })
                continue

            entries.append(entry)

    catalog = {
        "version": "1.0",
        "entries": entries,
        "build_report": {
            "total_runbooks": len(runbook_data),
            "mapped_entries": len(entries),
            "unmatched_runbooks": len(unmatched_runbooks),
            "empty_signal_runbooks": len(empty_signal_runbooks),
            "multi_mapping_runbooks": len(multi_mapping_runbooks)
        }
    }

    report = {
        "total_runbooks": len(runbook_data),
        "mapped_entries": len(entries),
        "unmatched_runbooks": unmatched_runbooks,
        "empty_signal_runbooks": empty_signal_runbooks,
        "multi_mapping_runbooks": multi_mapping_runbooks
    }

    save_json(catalog, OUTPUT_JSON)
    save_json(report, REPORT_JSON)

    print(f"✅ DONE: {len(entries)} keyword catalog entries")
    print(f"⚠️ Unmatched runbooks: {len(unmatched_runbooks)}")
    print(f"⚠️ Empty signal runbooks: {len(empty_signal_runbooks)}")
    print(f"⚠️ Multi-mapping runbooks: {len(multi_mapping_runbooks)}")
    print(f"📁 Saved catalog: {OUTPUT_JSON}")
    print(f"📁 Saved report: {REPORT_JSON}")

    if entries:
        print("\n===== SAMPLE ENTRY =====")
        print(json.dumps(entries[0], ensure_ascii=False, indent=2))


# ========================
# RUN
# ========================

if __name__ == "__main__":
    build_keyword_catalog()