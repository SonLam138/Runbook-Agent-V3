import json
from pathlib import Path


SERVICE_MAPPING_PATH = Path(r"D:\Runbook_Agent\V4_nho\service_mapping.json")


PATCHES = {
    "6.01-Thư điện tử": [
        "Exchange 2016",
        "Microsoft Exchange",
        "Mail Exchange",
        "ActiveDirectory",
        "AD",
        "ADManagerPlus",
        "EntraID"
    ],

    "3.04-Quản lý tài khoản ứng dụng": [
        "ActiveDirectory",
        "AD",
        "ADManagerPlus",
        "EntraID"
    ],

    "6.13-Microsoft Team / Office 365": [
        "Microsoft Exchange",
        "Exchange 2016",
        "Mail Exchange",
        "EntraID"
    ],

    "6.04-Chat Lync": [
        "Lync Server 2013"
    ],

    "6.12-Hội nghị truyền hình": [
        "Lync Server 2013",
        "OnePV"
    ],

    "8.02-Hạ tầng IT tại Hội sở": [
        "8.02 Hạ tầng tại hội sở",
        "DHCP",
        "IIS",
        "Windows Server, SQL"
    ],

    "99.02-Máy chủ (K.CNTT)": [
        "IIS",
        "DHCP",
        "Windows Server, SQL"
    ],

    "99.01-Database": [
        "Windows Server, SQL",
        "SQL",
        "SQL Server"
    ],

    "99.04-Network (K.CNTT)": [
        "DHCP",
        "8.02 Hạ tầng tại hội sở"
    ]
}


def add_unique(existing, additions):
    seen = {str(x).lower().strip() for x in existing}
    output = list(existing)

    for item in additions:
        key = str(item).lower().strip()
        if key not in seen:
            seen.add(key)
            output.append(item)

    return output


def main():
    with open(SERVICE_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for service_id, additions in PATCHES.items():
        if service_id not in mapping:
            print(f"⚠️ Missing service_id in mapping: {service_id}")
            continue

        old_list = mapping[service_id].get("technical_services", [])
        mapping[service_id]["technical_services"] = add_unique(old_list, additions)
        mapping[service_id]["mapping_status"] = "baseline_plus_manual_patch"

        print(f"✅ Patched {service_id}: +{len(additions)} candidates")

    with open(SERVICE_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved patched mapping → {SERVICE_MAPPING_PATH}")


if __name__ == "__main__":
    main()