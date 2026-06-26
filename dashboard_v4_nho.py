
# dashboard_v4_nho.py
# Streamlit dashboard for V4_nho email-level observability, catalog enrichment, and training dataset review.
# Run locally:
#   streamlit run dashboard_v4_nho.py

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd


# =========================
# CONFIG
# =========================

DEFAULT_LOG_PATH = Path("logs_v4_nho/email_summary.jsonl")
DEFAULT_EVENTS_PATH = Path("logs_v4_nho/events.jsonl")

SUCCESS_STATUSES = {"success", "done"}
SERVICE_LAYER_STATUSES = {"need_llm_fallback", "no_service", "service_not_resolved"}
SERVICE_LAYER_DECISIONS = {
    "no_service_signal",
    "low_confidence",
    "ambiguous_top_candidates",
    "cross_domain_conflict",
}
ISSUE_SCOPE_DECISIONS = {"empty_issue_scope_for_service"}
ISSUE_MATCHING_DECISION_PREFIXES = ("weak_", "ambiguous_")
ISSUE_MATCHING_DECISIONS = {
    "field_priority_not_resolved",
    "llm_no_candidate",
    "llm_invalid_json",
    "llm_exception",
    "llm_call_failed",
    "llm_invalid_candidate_id",
    "llm_candidate_out_of_scope",
}


# =========================
# BASIC HELPERS
# =========================

def safe_get(obj: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        if key not in cur or cur.get(key) is None:
            return default
        cur = cur.get(key)
    return cur


def short_text(text: Any, limit: int = 220) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def json_dumps(data: Any, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent, default=str)


def parse_jsonl_text(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception as e:
            records.append({
                "email_id": f"invalid_line_{line_no}",
                "parse_error": str(e),
                "raw": line,
                "input": {},
                "resolution": {"service": {}, "issue": {}, "runbook": {}},
                "outcome": {"status": "parse_error", "reason": str(e), "error": str(e)},
                "decision_trace": [],
            })
    return records


def load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return parse_jsonl_text(path.read_text(encoding="utf-8"))


def sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: str(r.get("started_at") or r.get("finished_at") or r.get("email_id") or ""),
        reverse=True,
    )


# =========================
# CLASSIFICATION
# =========================

def classify_failure_layer(record: Dict[str, Any]) -> str:
    status = safe_get(record, ["outcome", "status"], "unknown")
    service_decision = safe_get(record, ["resolution", "service", "decision"])
    issue_decision = safe_get(record, ["resolution", "issue", "decision"])
    runbook_title = (
        safe_get(record, ["resolution", "runbook", "title"])
        or safe_get(record, ["resolution", "runbook", "selected_runbook_title"])
    )

    if status in SUCCESS_STATUSES:
        return "success"

    if status in {"error", "parse_error"}:
        return "runtime_error"

    if status in SERVICE_LAYER_STATUSES:
        return "service_layer"

    if service_decision in SERVICE_LAYER_DECISIONS:
        return "service_layer"

    if issue_decision in ISSUE_SCOPE_DECISIONS:
        return "issue_scope_layer"

    if isinstance(issue_decision, str):
        if issue_decision.startswith(ISSUE_MATCHING_DECISION_PREFIXES):
            return "issue_matching_layer"
        if issue_decision in ISSUE_MATCHING_DECISIONS:
            return "issue_matching_layer"

    if status == "no_runbook":
        return "runbook_lookup_layer"

    if status not in SUCCESS_STATUSES and not runbook_title:
        return "runbook_lookup_layer"

    return "unknown"


def training_label(record: Dict[str, Any]) -> str:
    layer = classify_failure_layer(record)
    status = safe_get(record, ["outcome", "status"], "unknown")

    if layer == "success" or status in SUCCESS_STATUSES:
        return "positive_sample"
    if layer == "service_layer":
        return "needs_service_catalog_enrichment"
    if layer == "issue_scope_layer":
        return "needs_keyword_catalog_new_entry"
    if layer == "issue_matching_layer":
        return "needs_keyword_pattern_tuning"
    if layer == "runbook_lookup_layer":
        return "needs_runbook_mapping_fix"
    if layer == "runtime_error":
        return "needs_code_or_input_fix"
    return "needs_manual_review"


def enrich_suggestion(record: Dict[str, Any]) -> Dict[str, str]:
    status = safe_get(record, ["outcome", "status"], "unknown")
    layer = classify_failure_layer(record)
    subject = safe_get(record, ["input", "subject"], "") or ""
    service_id = (
        safe_get(record, ["resolution", "service", "service_id"])
        or safe_get(record, ["resolution", "service", "selected_service_id"])
        or ""
    )
    service_decision = safe_get(record, ["resolution", "service", "decision"])
    issue_decision = safe_get(record, ["resolution", "issue", "decision"])
    best_title = safe_get(record, ["resolution", "issue", "best_candidate", "title"])
    matched_text = safe_get(record, ["resolution", "issue", "best_candidate", "matched_text"])

    if layer == "service_layer":
        return {
            "layer": "Service catalog",
            "action": "Rà service aliases / hard anchors",
            "detail": f"Service chưa ổn định. Decision={service_decision}. Kiểm tra top_k, loại keyword generic và bổ sung alias/anchor cho service đúng. Subject='{subject}'.",
        }

    if layer == "issue_scope_layer":
        return {
            "layer": "Keyword catalog",
            "action": "Tạo entry đầu tiên cho service scope",
            "detail": f"Service '{service_id}' chưa có issue scope. Tạo keyword_catalog entry mới từ subject/body/OCR của email này.",
        }

    if layer == "issue_matching_layer":
        return {
            "layer": "Keyword enrichment",
            "action": "Bổ sung keyword_list / intents / patterns",
            "detail": f"Issue decision={issue_decision}. Best candidate={best_title or 'none'}, matched_text={matched_text or 'none'}. Thêm phrase đặc trưng hơn hoặc term phân biệt nếu ambiguous.",
        }

    if layer == "runbook_lookup_layer":
        return {
            "layer": "Runbook mapping",
            "action": "Kiểm tra title/source_file giữa keyword_catalog và runbook_data",
            "detail": "Service/issue có thể đã đi tiếp nhưng runbook content chưa attach được. Kiểm tra title trong keyword_catalog có khớp title trong runbook_data.json không.",
        }

    if layer == "runtime_error":
        return {
            "layer": "Runtime/Input",
            "action": "Fix code/path/input",
            "detail": f"Outcome={status}. Xem outcome.error và decision_trace để sửa lỗi runtime hoặc file input.",
        }

    if layer == "success":
        return {
            "layer": "Training dataset",
            "action": "Gắn positive sample",
            "detail": "Case đã pass end-to-end. Có thể đưa vào tập positive/evaluation sample.",
        }

    return {
        "layer": "Manual review",
        "action": "Review thủ công",
        "detail": "Không phân loại được. Xem raw summary và events nếu cần.",
    }


# =========================
# AGGREGATION
# =========================

def count_by(records: List[Dict[str, Any]], getter) -> pd.DataFrame:
    c = Counter()
    for r in records:
        key = getter(r)
        if key is None or key == "":
            key = "null_or_unknown"
        c[str(key)] += 1
    return pd.DataFrame([{"name": k, "count": v} for k, v in c.most_common()])


def avg_numeric(records: List[Dict[str, Any]], getter) -> Optional[float]:
    vals = []
    for r in records:
        v = getter(r)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def build_case_rows(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "email_id": r.get("email_id"),
            "started_at": r.get("started_at"),
            "subject": safe_get(r, ["input", "subject"]),
            "outcome": safe_get(r, ["outcome", "status"], "unknown"),
            "failure_layer": classify_failure_layer(r),
            "training_label": training_label(r),
            "service_id": safe_get(r, ["resolution", "service", "service_id"]),
            "service_decision": safe_get(r, ["resolution", "service", "decision"]),
            "service_confidence": safe_get(r, ["resolution", "service", "confidence"]),
            "issue_decision": safe_get(r, ["resolution", "issue", "decision"]),
            "issue_flow": safe_get(r, ["resolution", "issue", "resolver_flow"]),
            "issue_confidence": safe_get(r, ["resolution", "issue", "confidence"]),
            "best_candidate": safe_get(r, ["resolution", "issue", "best_candidate", "title"]),
            "runbook_title": safe_get(r, ["resolution", "runbook", "title"]),
            "runbook_source": safe_get(r, ["resolution", "runbook", "source_file"]),
        })
    return pd.DataFrame(rows)


def build_training_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        suggestion = enrich_suggestion(r)
        out.append({
            "email_id": r.get("email_id"),
            "started_at": r.get("started_at"),
            "subject": safe_get(r, ["input", "subject"]),
            "body_text": safe_get(r, ["input", "body_text"]),
            "ocr_text": safe_get(r, ["input", "ocr_text"]),
            "final_email_text": safe_get(r, ["input", "final_email_text"]),
            "outcome_status": safe_get(r, ["outcome", "status"]),
            "failure_layer": classify_failure_layer(r),
            "training_label": training_label(r),
            "resolved_service_id": safe_get(r, ["resolution", "service", "service_id"]),
            "service_decision": safe_get(r, ["resolution", "service", "decision"]),
            "service_confidence": safe_get(r, ["resolution", "service", "confidence"]),
            "issue_decision": safe_get(r, ["resolution", "issue", "decision"]),
            "issue_flow": safe_get(r, ["resolution", "issue", "resolver_flow"]),
            "issue_confidence": safe_get(r, ["resolution", "issue", "confidence"]),
            "best_candidate_title": safe_get(r, ["resolution", "issue", "best_candidate", "title"]),
            "best_candidate_matched_field": safe_get(r, ["resolution", "issue", "best_candidate", "matched_field"]),
            "best_candidate_matched_text": safe_get(r, ["resolution", "issue", "best_candidate", "matched_text"]),
            "runbook_title": safe_get(r, ["resolution", "runbook", "title"]),
            "runbook_source_file": safe_get(r, ["resolution", "runbook", "source_file"]),
            "suggested_layer": suggestion["layer"],
            "suggested_action": suggestion["action"],
            "suggested_detail": suggestion["detail"],
        })
    return out


def to_jsonl(records: List[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records)


# =========================
# UI COMPONENTS
# =========================

def render_metric_cards(records: List[Dict[str, Any]]) -> None:
    total = len(records)
    success = sum(1 for r in records if classify_failure_layer(r) == "success")
    service_layer = sum(1 for r in records if classify_failure_layer(r) == "service_layer")
    issue_layer = sum(1 for r in records if classify_failure_layer(r) in {"issue_scope_layer", "issue_matching_layer"})
    runbook_layer = sum(1 for r in records if classify_failure_layer(r) == "runbook_lookup_layer")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total emails", total)
    c2.metric("Success", success, f"{round(success / total * 100, 1) if total else 0}%")
    c3.metric("Service fail", service_layer)
    c4.metric("Issue fail", issue_layer)
    c5.metric("Runbook fail", runbook_layer)


def render_record_summary(record, prefix="latest"):

    email_id = record.get("email_id", "unknown")

    ...

    with st.expander("Input email", expanded=True):

        st.markdown(f"**Subject:** {safe_get(record, ['input', 'subject'], '(empty)') or '(empty)'}")

        c1, c2 = st.columns(2)

        c1.text_area(
            "Body",
            value=safe_get(record, ["input", "body_text"], "") or "",
            height=160,
            key=f"{prefix}_body_{email_id}"
        )

        c2.text_area(
            "OCR",
            value=safe_get(record, ["input", "ocr_text"], "") or "",
            height=160,
            key=f"{prefix}_ocr_{email_id}"
        )

        st.text_area(
            "Final email text",
            value=safe_get(record, ["input", "final_email_text"], "") or "",
            height=160,
            key=f"{prefix}_final_{email_id}"
        )

    service = safe_get(record, ["resolution", "service"], {}) or {}
    issue = safe_get(record, ["resolution", "issue"], {}) or {}
    runbook = safe_get(record, ["resolution", "runbook"], {}) or {}

    svc_col, issue_col, rb_col = st.columns(3)

    with svc_col:
        st.markdown("### 🎯 Service")
        st.write({
            "service_id": service.get("service_id") or service.get("selected_service_id"),
            "service_name": service.get("service_name") or service.get("selected_service_name"),
            "decision": service.get("decision"),
            "confidence": service.get("confidence"),
            "need_llm_fallback": service.get("need_llm_fallback"),
        })
        with st.expander("Service top_k / evidence"):
            st.json(service.get("top_k", []))

    with issue_col:
        st.markdown("### 🔎 Issue")
        st.write({
            "resolved": issue.get("resolved"),
            "decision": issue.get("decision"),
            "confidence": issue.get("confidence"),
            "resolver_flow": issue.get("resolver_flow"),
            "llm_used": issue.get("llm_used"),
        })
        with st.expander("Best candidate"):
            st.json(issue.get("best_candidate"))
        with st.expander("Field trace"):
            st.json(issue.get("field_trace", []))
        with st.expander("Issue top_k"):
            st.json(issue.get("top_k", []))

    with rb_col:
        st.markdown("### 📘 Runbook")
        st.write({
            "title": runbook.get("title") or runbook.get("selected_runbook_title"),
            "service": runbook.get("service"),
            "keyword": runbook.get("keyword"),
            "source_file": runbook.get("source_file") or runbook.get("selected_runbook_id"),
        })

    st.markdown("### 🧭 Decision trace")
    trace = record.get("decision_trace", []) or []
    if trace:
        st.dataframe(pd.DataFrame(trace), use_container_width=True)
    else:
        st.caption("Không có decision_trace.")

    st.markdown("### 🧪 Enrich suggestion")
    suggestion = enrich_suggestion(record)
    st.info(f"**{suggestion['layer']} → {suggestion['action']}**\n\n{suggestion['detail']}")
    st.markdown(f"**Training label đề xuất:** `{training_label(record)}`")


def render_aggregate(records: List[Dict[str, Any]]) -> None:
    st.subheader("📊 Tổng hợp toàn luồng")

    if not records:
        st.info("Không có dữ liệu để phân tích.")
        return

    outcome_df = count_by(records, lambda r: safe_get(r, ["outcome", "status"], "unknown"))
    layer_df = count_by(records, classify_failure_layer)
    service_decision_df = count_by(records, lambda r: safe_get(r, ["resolution", "service", "decision"], "not_started"))
    issue_decision_df = count_by(records, lambda r: safe_get(r, ["resolution", "issue", "decision"], "not_started"))
    issue_flow_df = count_by(records, lambda r: safe_get(r, ["resolution", "issue", "resolver_flow"], "null_or_not_started"))
    service_df = count_by(records, lambda r: safe_get(r, ["resolution", "service", "service_id"], "unknown_service"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Outcome distribution")
        st.bar_chart(outcome_df.set_index("name"))
    with c2:
        st.markdown("#### Failure layer")
        st.bar_chart(layer_df.set_index("name"))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Service decisions")
        st.dataframe(service_decision_df, use_container_width=True)
        st.markdown(f"Average service confidence: `{avg_numeric(records, lambda r: safe_get(r, ['resolution', 'service', 'confidence']))}`")
    with c4:
        st.markdown("#### Issue decisions")
        st.dataframe(issue_decision_df, use_container_width=True)
        st.markdown("#### Issue flow")
        st.dataframe(issue_flow_df, use_container_width=True)
        st.markdown(f"Average issue confidence: `{avg_numeric(records, lambda r: safe_get(r, ['resolution', 'issue', 'confidence']))}`")

    st.markdown("#### Top services")
    st.dataframe(service_df, use_container_width=True)


def render_cases(records: List[Dict[str, Any]]) -> None:
    st.subheader("🗂️ Case review")

    if not records:
        st.info("Không có dữ liệu.")
        return

    df = build_case_rows(records)

    layers = ["all"] + sorted(df["failure_layer"].dropna().unique().tolist())
    outcomes = ["all"] + sorted(df["outcome"].dropna().unique().tolist())
    labels = ["all"] + sorted(df["training_label"].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    layer_filter = c1.selectbox("Failure layer", layers)
    outcome_filter = c2.selectbox("Outcome", outcomes)
    label_filter = c3.selectbox("Training label", labels)
    search = c4.text_input("Search subject/email_id/service", "")

    filtered = df.copy()
    if layer_filter != "all":
        filtered = filtered[filtered["failure_layer"] == layer_filter]
    if outcome_filter != "all":
        filtered = filtered[filtered["outcome"] == outcome_filter]
    if label_filter != "all":
        filtered = filtered[filtered["training_label"] == label_filter]
    if search.strip():
        s = search.lower().strip()
        filtered = filtered[
            filtered.apply(lambda row: s in " ".join(str(x).lower() for x in row.values), axis=1)
        ]

    st.dataframe(filtered, use_container_width=True, height=360)

    if not filtered.empty:
        selected_email = st.selectbox("Chọn case để xem chi tiết", filtered["email_id"].tolist())
        selected = next((r for r in records if r.get("email_id") == selected_email), None)
        if selected:
            render_record_summary(selected, prefix="case")


def render_training_export(records: List[Dict[str, Any]]) -> None:
    st.subheader("🎓 Training export")

    training_records = build_training_records(records)
    df = pd.DataFrame(training_records)

    if df.empty:
        st.info("Không có dữ liệu export.")
        return

    label_filter = st.multiselect(
        "Chọn training_label để export",
        options=sorted(df["training_label"].dropna().unique().tolist()),
        default=sorted(df["training_label"].dropna().unique().tolist()),
    )

    export_df = df[df["training_label"].isin(label_filter)] if label_filter else df.iloc[0:0]

    st.dataframe(export_df, use_container_width=True, height=360)

    jsonl_data = to_jsonl(export_df.to_dict(orient="records"))
    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download training_dataset.jsonl",
        data=jsonl_data.encode("utf-8"),
        file_name="training_dataset_v4_nho.jsonl",
        mime="application/jsonl",
    )
    c2.download_button(
        "Download training_dataset.csv",
        data=csv_data,
        file_name="training_dataset_v4_nho.csv",
        mime="text/csv",
    )

    st.markdown("#### Label distribution")
    st.bar_chart(count_by(export_df.to_dict(orient="records"), lambda r: r.get("training_label")).set_index("name"))


def render_raw(records: List[Dict[str, Any]]) -> None:
    st.subheader("🧾 Raw JSON")
    selected = st.selectbox("Chọn record", [r.get("email_id", f"record_{i}") for i, r in enumerate(records)]) if records else None
    if selected:
        record = next((r for r in records if r.get("email_id") == selected), records[0])
        st.json(record)


# =========================
# MAIN APP
# =========================

def main() -> None:
    st.set_page_config(
        page_title="V4_nho Learning Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 V4_nhỏ Learning Dashboard")
    st.caption("Catalog Enrichment Workbench + Resolver Debug Dashboard + Training Dataset Builder")

    with st.sidebar:
        st.header("Data source")
        log_path_text = st.text_input("email_summary.jsonl path", value=str(DEFAULT_LOG_PATH))
        log_path = Path(log_path_text)

        mode = st.radio(
            "Load mode",
            ["Read file", "Paste JSONL"],
            index=0,
        )

        pasted_jsonl = ""
        if mode == "Paste JSONL":
            pasted_jsonl = st.text_area("Paste JSONL", key="paste_jsonl")

        if st.button("Refresh / Re-read"):
            st.cache_data.clear()

        st.divider()
        st.markdown("**Nguồn chuẩn:** `email_summary.jsonl`")
        st.markdown("**Debug sâu:** `events.jsonl`")

    if mode == "Read file":
        records = load_jsonl_file(log_path)
        if not records:
            st.warning(f"Không đọc được record nào từ: {log_path}")
    else:
        records = parse_jsonl_text(pasted_jsonl)
        if not records:
            st.info("Dán JSONL để bắt đầu phân tích.")

    records = sort_records(records)

    render_metric_cards(records)

    tab_latest, tab_aggregate, tab_cases, tab_training, tab_raw = st.tabs([
        "Latest Email Review",
        "Aggregate Analysis",
        "Case Review",
        "Training Export",
        "Raw JSON",
    ])

    with tab_latest:
        latest = records[0] if records else {}
        render_record_summary(latest, prefix="latest")

    with tab_aggregate:
        render_aggregate(records)

    with tab_cases:
        render_cases(records)

    with tab_training:
        render_training_export(records)

    with tab_raw:
        render_raw(records)

    st.divider()
    st.caption(
        "Design note: email_summary.jsonl là source of truth cho phân tích end-to-end; "
        "events.jsonl dùng khi cần drill-down sâu theo từng stage."
    )


if __name__ == "__main__":
    main()
