from email_processor_V4_nho import process_eml_file
from service_resolver_V4_nho import service_resolver_v2
from issue_resolver_V4_nho import issue_resolver_V4_nho
from logs_v4_nho.logger_v4_nho import V4NhoLogger


def run_v4_nho(eml_path: str, email_id: str | None = None):
    logger = V4NhoLogger(email_id=email_id)

    try:
        # ========================
        # 1. Process email
        # ========================
        email_result = process_eml_file(eml_path)

        email_text = email_result.get("email_text", "")
        subject = email_result.get("subject")
        body_text = email_result.get("body_text")
        ocr_text = email_result.get("ocr_text")

        # ✅ LOG: email input (rất quan trọng cho debug + training)
        logger.set_email_input(
            subject=subject,
            body_text=body_text,
            ocr_text=ocr_text,
            final_email_text=email_text,
        )

        # ========================
        # 2. Resolve service
        # ========================
        service_result = service_resolver_v2(email_text)

        # ✅ LOG: service resolver
        logger.set_service_resolution(service_result)

        # ✅ DECISION BRANCHES (giữ nguyên logic, chỉ thêm finalize)
        if service_result.get("need_llm_fallback"):
            return {
                "status": "need_llm_fallback",
                "email_text": email_text,
                "log_summary": logger.finalize(
                    status="need_llm_fallback",
                    reason="Service resolver requires LLM fallback."
                )
            }

        if not service_result.get("resolved"):
            return {
                "status": "service_not_resolved",
                "email_text": email_text,
                "log_summary": logger.finalize(
                    status="no_service",
                    reason="Service resolver could not resolve service."
                )
            }

        # Optional: nếu muốn bắt conflict (khuyên bật)
        conflict = service_result.get("conflict")
        if isinstance(conflict, dict) and conflict.get("has_conflict"):
            return {
                "status": "service_conflict",
                "service": service_result,
                "log_summary": logger.finalize(
                    status="service_conflict",
                    reason="Multiple services detected with similar confidence."
                )
            }

        resolved_service_id = service_result.get("service_id")

        # ========================
        # 3. Resolve issue
        # ========================
        issue_result = issue_resolver_V4_nho(
            email_text=email_text,
            resolved_service_id=resolved_service_id
        )

        # ✅ LOG: issue resolver
        logger.set_issue_resolution(issue_result)

        if not issue_result.get("runbook_found"):
            return {
                "status": "no_runbook",
                "service": service_result,
                "email_text": email_text,
                "log_summary": logger.finalize(
                    status="no_runbook",
                    reason="No runbook found for resolved service."
                )
            }

        # ========================
        # 4. Final output (SUCCESS)
        # ========================
        summary = logger.finalize(
            status="success",
            reason="Email processed successfully and runbook selected."
        )

        return {
            "status": "done",
            "service": service_result,
            "issue": issue_result,
            "email_preview": email_text[:1000],  # debug
            "log_summary": summary   # ✅ trả log luôn để test
        }

    except Exception as e:
        # ✅ LOG: unexpected error
        summary = logger.finalize(
            status="error",
            reason="Unhandled exception in run_v4_nho.",
            error=str(e),
            extra={"eml_path": eml_path}
        )

        return {
            "status": "error",
            "error": str(e),
            "log_summary": summary
        }