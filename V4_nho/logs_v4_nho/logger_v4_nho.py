# logger_v4_nho.py
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path("logs_v4_nho")
EVENT_LOG_FILE = LOG_DIR / "events.jsonl"
SUMMARY_LOG_FILE = LOG_DIR / "email_summary.jsonl"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def generate_email_id(prefix: str = "email") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short_id}"


def safe_json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def append_jsonl(file_path: Path, record: Dict[str, Any]) -> None:
    ensure_log_dir()
    with file_path.open("a", encoding="utf-8") as f:
        f.write(safe_json_dumps(record) + "\n")


class V4NhoLogger:
    def __init__(
        self,
        email_id: Optional[str] = None,
        source: str = "servicedesk_email",
        enable_event_log: bool = True,
        enable_summary_log: bool = True,
    ):
        self.email_id = email_id or generate_email_id()
        self.source = source
        self.enable_event_log = enable_event_log
        self.enable_summary_log = enable_summary_log

        self.started_at = now_iso()
        self.events: List[Dict[str, Any]] = []
        self.decision_trace: List[Dict[str, Any]] = []

        self.summary: Dict[str, Any] = {
            "email_id": self.email_id,
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": None,

            "input": {
                "subject": None,
                "body_text": None,
                "ocr_text": None,
                "final_email_text": None,
            },

            "resolution": {
                "service": {},
                "issue": {},
                "runbook": {},
            },

            "decision_trace": self.decision_trace,

            "outcome": {
                "status": None,
                "reason": None,
                "error": None,
            },
        }

    # =========================================
    # CORE LOG
    # =========================================
    def log_event(self, event: str, stage: str, data: Optional[Dict] = None):
        record = {
            "timestamp": now_iso(),
            "email_id": self.email_id,
            "event": event,
            "stage": stage,
            "data": data or {},
        }
        self.events.append(record)
        if self.enable_event_log:
            append_jsonl(EVENT_LOG_FILE, record)

    def add_decision(self, action: str, reason=None, confidence=None, details=None):
        self.decision_trace.append({
            "timestamp": now_iso(),
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "details": details or {},
        })

    # =========================================
    # EMAIL INPUT
    # =========================================
    def set_email_input(self, subject, body_text, ocr_text, final_email_text):
        self.summary["input"].update({
            "subject": subject,
            "body_text": body_text,
            "ocr_text": ocr_text,
            "final_email_text": final_email_text,
        })

        self.log_event(
            "email_text_built",
            "email_processor",
            {
                "len": len(final_email_text or "")
            }
        )

    # =========================================
    # SERVICE RESOLUTION
    # =========================================
    def set_service_resolution(self, service_result: dict):
        data = {
            "service_id": service_result.get("service_id"),
            "service_name": service_result.get("service_name"),
            "confidence": service_result.get("confidence"),
            "decision": service_result.get("decision"),
            "need_llm_fallback": service_result.get("need_llm_fallback"),
            "conflict": service_result.get("conflict"),
            "top_k": service_result.get("top_k", []),
        }

        self.summary["resolution"]["service"] = data

        self.log_event("service_resolved", "service_resolver", data)

        self.add_decision(
            "service_resolved",
            reason=data["decision"],
            confidence=data["confidence"]
        )

    # =========================================
    # ISSUE + RUNBOOK
    # =========================================
    def set_issue_resolution(self, issue_result: dict) -> None:
        """
        Log full kết quả issue_resolver theo style V4.
        """

        best = issue_result.get("best_candidate") or {}
        resolved = issue_result.get("resolved")

        issue_data = {
            "resolved": resolved,
            "decision": issue_result.get("decision"),
            "confidence": issue_result.get("confidence"),
            "resolver_flow": issue_result.get("resolver_flow"),

            "best_candidate": {
                "title": best.get("title"),
                "score": best.get("score"),
                "confidence": best.get("confidence"),
                "matched_field": best.get("matched_field"),
                "matched_text": best.get("matched_text"),
                "match_type": best.get("match_type"),
                "overlap_tokens": best.get("overlap_tokens", []),
            } if best else None,

            "top_k": issue_result.get("top_k", []),
            "field_trace": issue_result.get("field_trace", []),
            "llm_used": issue_result.get("resolver_flow") == "llm_fallback"
        }

        # ✅ attach runbook (nếu có)
        runbook = issue_result.get("runbook")

        runbook_data = {
            "title": runbook.get("title") if runbook else None,
            "service": runbook.get("service") if runbook else None,
            "keyword": runbook.get("keyword") if runbook else None,
            "source_file": runbook.get("source_file") if runbook else None,
        }

        self.summary["resolution"]["issue"] = issue_data
        self.summary["resolution"]["runbook"] = runbook_data

        # ✅ event log
        self.log_event(
            event="issue_resolved",
            stage="issue_resolver",
            data={
                "resolved": resolved,
                "decision": issue_result.get("decision"),
                "confidence": issue_result.get("confidence"),
                "resolver_flow": issue_result.get("resolver_flow"),
                "best_title": best.get("title"),
            }
        )

        # ✅ decision trace
        self.add_decision(
            action="issue_resolved",
            reason=issue_result.get("decision"),
            confidence=issue_result.get("confidence"),
            details={
                "resolver_flow": issue_result.get("resolver_flow"),
                "best_candidate": best.get("title"),
                "match_type": best.get("match_type"),
                "matched_field": best.get("matched_field")
            }
        )
    # =========================================
    # FINALIZE
    # =========================================
    def finalize(self, status, reason=None, error=None, extra=None):
        self.summary["finished_at"] = now_iso()

        self.summary["outcome"] = {
            "status": status,
            "reason": reason,
            "error": error
        }

        if extra:
            self.summary["extra"] = extra

        self.log_event(
            "email_processing_finished",
            "run_v4_nho",
            {
                "status": status,
                "error": error
            }
        )

        self.add_decision("finalize", reason=reason)

        if self.enable_summary_log:
            append_jsonl(SUMMARY_LOG_FILE, self.summary)

        return self.summary