# test_issue_logging.py

from issue_resolver_V4_nho import issue_resolver_V4_nho
from logs_v4_nho.logger_v4_nho import V4NhoLogger


def build_test_email(subject: str, body: str, ocr: str = "") -> str:
    parts = []

    if subject:
        parts.append(f"Subject: {subject.strip()}")

    if body:
        parts.append(f"Body:\n{body.strip()}")

    if ocr:
        parts.append(f"OCR_Text_From_Inline_Image:\n{ocr.strip()}")

    return "\n\n".join(parts)


# =========================
# TEST INPUT (REALISTIC)
# =========================

subject = "Tạo tài khoản mới"

body = """
Kính gửi: Khối CNTT;

Kính nhờ Anh/chị hỗ trợ tạo tài khoản cho nhân viên mới
Họ và tên : A
Khối : CNTT

"""

ocr = """
Transact Sign in

username and password are case sensitive
please check your login credential
"""

email_text = build_test_email(subject, body, ocr)

resolved_service_id = "0.01-T24"


# =========================
# RUN ISSUE RESOLVER
# =========================

issue_result = issue_resolver_V4_nho(
    email_text=email_text,
    resolved_service_id=resolved_service_id,
    use_llm_fallback=False,   # test deterministic cho rõ
    attach_runbook=True
)


# =========================
# LOGGING TEST
# =========================

logger = V4NhoLogger()

# giả lập service đã resolved (để đầy đủ summary)
logger.summary["resolution"]["service"] = {
    "service_id": resolved_service_id,
    "decision": "resolved_for_test"
}

logger.set_issue_resolution(issue_result)

summary = logger.finalize(
    status="test_issue_logging",
    reason="Standalone issue logging test"
)


# =========================
# OUTPUT
# =========================
import json

print("\n===== ISSUE RESULT =====")
print(json.dumps(issue_result, indent=2, ensure_ascii=False))

print("\n===== LOG SUMMARY =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))