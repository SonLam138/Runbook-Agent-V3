from Preprocess_email import extract_clean_text_from_email
from service_resolver_V4_nho import service_resolver_v2
from Issue_tpye_resovler_V4_nho import resolve_issue_v4_nho

import json


# =====================================================
# TEST CASES
# =====================================================

test_cases = [
    {
        "name": "email_login_error",
        "email": {
            "subject": "Không đăng nhập được Outlook",
            "body": "Tôi nhập mật khẩu nhưng không vào được email"
        }
    },
    {
        "name": "vpn_connection_error",
        "email": {
            "subject": "VPN không kết nối",
            "body": "Không vào được mạng nội bộ"
        }
    },
    {
        "name": "video_conf_free_text",
        "email": {
            "subject": "Không vào được họp",
            "body": "Sáng nay mình không vào được buổi họp trực tuyến"
        }
    },
    {
        "name": "mail_inside_meeting",
        "email": {
            "subject": "Không gửi được mail",
            "body": "Trong hội nghị tôi không gửi được mail"
        }
    },
    {
        "name": "unknown_issue",
        "email": {
            "subject": "Máy bị chậm",
            "body": "Mở ứng dụng bị lag"
        }
    }
]


# =====================================================
# LOAD CATALOG
# =====================================================

def load_knowledge_catalog(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "services" in data:
        return data["services"]

    raise ValueError("knowledge_catalog phải là list hoặc dict chứa 'services'")


def load_keyword_catalog(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError("keyword_catalog phải là list")


# =====================================================
# FULL TEST FLOW (SERVICE + ISSUE)
# =====================================================

def run_full_tests(knowledge_catalog, keyword_catalog):

    for case in test_cases:
        print("\n" + "=" * 60)
        print(f"TEST: {case['name']}")

        # ========================
        # 1. PREPROCESS
        # ========================
        clean_text = extract_clean_text_from_email(case["email"])

        print("CLEAN TEXT:", clean_text)

        # ========================
        # 2. SERVICE RESOLVER
        # ========================
        service_result = service_resolver_v2(
            clean_text=clean_text,
            knowledge_catalog=knowledge_catalog,
            top_k=3
        )

        print("\n----- SERVICE RESOLVER -----")
        print("SERVICE_ID:", service_result.get("service_id"))
        print("SERVICE_NAME:", service_result.get("service_name"))
        print("CONFIDENCE:", service_result.get("confidence"))
        print("FALLBACK:", service_result.get("need_llm_fallback"))

        print("\nSERVICE TOP_K:")
        for idx, c in enumerate(service_result.get("top_k", []), start=1):
            print(f"  {idx}. {c['service_id']} | score={c['score']}")

        # ========================
        # 3. ISSUE RESOLVER
        # ========================

        resolved_service = {
            "service_id": service_result.get("service_id"),
            "service_name": service_result.get("service_name")
        }

        issue_result = resolve_issue_v4_nho(
            clean_text=clean_text,
            resolved_service=resolved_service,
            keyword_catalog=keyword_catalog,
            enable_llm_fallback=True
        )

        print("\n----- ISSUE RESOLVER -----")
        candidate_issue = issue_result.get("issue_title")
        final_issue = candidate_issue if issue_result.get("resolved") else None

        print("ISSUE CANDIDATE:", candidate_issue)
        print("ISSUE FINAL:", final_issue)
        print("ISSUE CONFIDENCE:", issue_result.get("confidence"))
        print("ISSUE FALLBACK:", issue_result.get("need_llm_fallback"))
        print("ISSUE DECISION:", issue_result.get("decision"))
        print("SOURCE:", issue_result.get("source"))

        # ========================
        # 4. DEBUG (OPTIONAL BUT IMPORTANT)
        # ========================

        if issue_result.get("deterministic"):
            print("\nDET TOP_K:")
            for c in issue_result["deterministic"].get("top_k", []):
                print("   ", c["issue_title"], "| score =", c["score"])

        if issue_result.get("llm"):
            print("\nLLM RAW OUTPUT:")
            print(issue_result["llm"])

        # ========================
        # 5. QUICK RESULT
        # ========================
        if issue_result.get("resolved"):
            print("✅ FINAL: Issue resolved")
        else:
            print("⚠️ FINAL: Need fallback / search")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print(">>> LOAD CATALOG")

    knowledge_catalog = load_knowledge_catalog(
        r"D:\Runbook_Agent\Data\knowledge_catalog.json"
    )

    keyword_catalog = load_keyword_catalog(
        r"D:\Runbook_Agent\Data\keyword_catalog.json"
    )

    print(f">>> LOADED SERVICES: {len(knowledge_catalog)}")
    print(f">>> LOADED KEYWORDS: {len(keyword_catalog)}")

    run_full_tests(knowledge_catalog, keyword_catalog)