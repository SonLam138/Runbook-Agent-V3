import json
import re
from collections import defaultdict


# =====================================================
# NORMALIZE
# =====================================================

def normalize_key(text):
    """
    Normalize nhẹ để so sánh service name:
    - lowercase
    - trim
    - gom nhiều khoảng trắng
    """

    if not text:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


# =====================================================
# LOAD JSON
# =====================================================

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# EXTRACT TECHNICAL SERVICES FROM KEYWORD CATALOG
# =====================================================

def collect_runbook_technical_services(keyword_catalog):
    """
    Lấy toàn bộ technical service đang tồn tại trong keyword_catalog.

    keyword_catalog item hiện có dạng:
    {
        "title": "...",
        "service": "Active Directory",
        "keyword": "...",
        "keyword_list": [...],
        ...
    }

    Output:
    {
        normalized_service_name: {
            "raw_name": "...",
            "count": số runbook/issue thuộc service này,
            "titles": [...]
        }
    }
    """

    services = {}

    for item in keyword_catalog:
        service = item.get("service", "")
        title = item.get("title", "")

        service_norm = normalize_key(service)

        if not service_norm:
            continue

        if service_norm not in services:
            services[service_norm] = {
                "raw_name": service,
                "count": 0,
                "titles": []
            }

        services[service_norm]["count"] += 1

        if title:
            services[service_norm]["titles"].append(title)

    return services


# =====================================================
# CHECK ONE BUSINESS SERVICE MAPPING
# =====================================================

def check_one_service_mapping(service_id, mapping_item, technical_service_index):
    """
    Kiểm tra một business service có map được technical service nào
    trong keyword_catalog hay không.
    """

    service_name = mapping_item.get("service_name", "")
    technical_services = mapping_item.get("technical_services", []) or []

    matched = []
    missing = []

    for tech in technical_services:
        tech_norm = normalize_key(tech)

        if tech_norm in technical_service_index:
            matched.append({
                "mapping_value": tech,
                "matched_runbook_service": technical_service_index[tech_norm]["raw_name"],
                "runbook_count": technical_service_index[tech_norm]["count"],
                "sample_titles": technical_service_index[tech_norm]["titles"][:5]
            })
        else:
            missing.append(tech)

    has_match = len(matched) > 0

    return {
        "service_id": service_id,
        "service_name": service_name,
        "has_match": has_match,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "matched": matched,
        "missing": missing
    }


# =====================================================
# CHECK FULL SERVICE MAPPING
# =====================================================

def check_service_mapping(service_mapping, keyword_catalog):
    """
    Kiểm tra toàn bộ service_mapping.json với keyword_catalog.json.

    Trả ra report:
    - business services nào map được
    - business services nào chưa map được
    - technical services trong runbook chưa được business service nào cover
    """

    technical_service_index = collect_runbook_technical_services(keyword_catalog)

    report = {
        "summary": {
            "total_business_services": 0,
            "mapped_business_services": 0,
            "unmapped_business_services": 0,
            "total_runbook_technical_services": len(technical_service_index),
            "covered_runbook_technical_services": 0,
            "uncovered_runbook_technical_services": 0
        },
        "mapped_services": [],
        "unmapped_services": [],
        "uncovered_runbook_services": []
    }

    covered_technical_services = set()

    for service_id, mapping_item in service_mapping.items():
        report["summary"]["total_business_services"] += 1

        result = check_one_service_mapping(
            service_id=service_id,
            mapping_item=mapping_item,
            technical_service_index=technical_service_index
        )

        if result["has_match"]:
            report["summary"]["mapped_business_services"] += 1
            report["mapped_services"].append(result)

            for m in result["matched"]:
                covered_technical_services.add(
                    normalize_key(m["matched_runbook_service"])
                )
        else:
            report["summary"]["unmapped_business_services"] += 1
            report["unmapped_services"].append(result)

    # technical service trong keyword_catalog nhưng chưa được mapping nào cover
    for tech_norm, data in technical_service_index.items():
        if tech_norm not in covered_technical_services:
            report["uncovered_runbook_services"].append({
                "runbook_service": data["raw_name"],
                "runbook_count": data["count"],
                "sample_titles": data["titles"][:5]
            })

    report["summary"]["covered_runbook_technical_services"] = len(covered_technical_services)
    report["summary"]["uncovered_runbook_technical_services"] = len(
        report["uncovered_runbook_services"]
    )

    return report


# =====================================================
# PRINT SUMMARY
# =====================================================

def print_mapping_report(report):
    """
    In báo cáo gọn ra console.
    """

    summary = report["summary"]

    print("\n" + "=" * 70)
    print("SERVICE MAPPING CHECK SUMMARY")
    print("=" * 70)

    print(f"Total business services           : {summary['total_business_services']}")
    print(f"Mapped business services          : {summary['mapped_business_services']}")
    print(f"Unmapped business services        : {summary['unmapped_business_services']}")
    print(f"Total runbook technical services  : {summary['total_runbook_technical_services']}")
    print(f"Covered runbook technical services: {summary['covered_runbook_technical_services']}")
    print(f"Uncovered runbook technical svcs  : {summary['uncovered_runbook_technical_services']}")

    print("\n" + "-" * 70)
    print("UNMAPPED BUSINESS SERVICES")
    print("-" * 70)

    for item in report["unmapped_services"]:
        print(f"- {item['service_id']} | {item['service_name']}")

    print("\n" + "-" * 70)
    print("UNCOVERED RUNBOOK TECHNICAL SERVICES")
    print("-" * 70)

    for item in report["uncovered_runbook_services"]:
        print(f"- {item['runbook_service']} | runbooks={item['runbook_count']}")
        for t in item["sample_titles"]:
            print(f"    · {t}")


# =====================================================
# SAVE REPORT
# =====================================================

def save_mapping_report(report, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved mapping report → {output_path}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    SERVICE_MAPPING_PATH = r"D:\Runbook_Agent\Data\service_mapping.json"
    KEYWORD_CATALOG_PATH = r"D:\Runbook_Agent\Data\keyword_catalog.json"
    OUTPUT_REPORT_PATH = r"D:\Runbook_Agent\Data\service_mapping_report.json"

    service_mapping = load_json(SERVICE_MAPPING_PATH)
    keyword_catalog = load_json(KEYWORD_CATALOG_PATH)

    report = check_service_mapping(
        service_mapping=service_mapping,
        keyword_catalog=keyword_catalog
    )

    print_mapping_report(report)

    save_mapping_report(
        report=report,
        output_path=OUTPUT_REPORT_PATH
    )