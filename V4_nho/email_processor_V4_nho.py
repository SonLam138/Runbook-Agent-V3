import re
import io
import os
import json
import base64
import html
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage
import pytesseract

# ========================
# CONFIG
# ========================


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@dataclass
class EmailProcessorConfig:
    """
    Config cho email processor V4_nho.

    Scope hiện tại:
    - xử lý text body
    - xử lý inline image user paste trong body email
    - không ưu tiên attachment rời
    """

    enable_ocr: bool = True

    # Chỉ OCR image có vẻ là inline/pasted image trong body.
    # Attachment rời sẽ bị skip ở phase này.
    inline_images_only: bool = True

    # Filter ảnh nhỏ để tránh OCR logo/signature.
    min_image_width: int = 250
    min_image_height: int = 120
    min_image_area: int = 40000

    # OCR language.
    # Nếu máy chưa cài traineddata "vie", code sẽ fallback về "eng".
    ocr_lang: str = "eng+vie"

    # Có giữ OCR text không chắc liên quan lỗi không?
    # False = chỉ append OCR có keyword lỗi / đủ signal.
    keep_low_signal_ocr: bool = False

    # Giới hạn độ dài OCR text append vào resolver.
    max_ocr_text_length: int = 2000

    # Có remove chữ ký và email chain không?
    remove_signature: bool = True
    remove_chain: bool = True


DEFAULT_CONFIG = EmailProcessorConfig()


# ========================
# HIGH LEVEL API
# ========================

def process_eml_file(
    eml_path: str,
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Entry point chính khi có file .eml.

    Output:
    {
      "subject": "...",
      "body_text": "...",
      "ocr_text": "...",
      "email_text": "...",
      "diagnostics": {...}
    }
    """

    msg = parse_eml_file(eml_path)

    return process_email_message(
        msg=msg,
        config=config
    )


def process_eml_bytes(
    raw_bytes: bytes,
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Entry point nếu email lấy từ API/connector và có raw MIME bytes.
    """

    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    return process_email_message(
        msg=msg,
        config=config
    )


def process_raw_email_fields(
    subject: str,
    body_text: str = "",
    body_html: str = "",
    inline_image_bytes: Optional[List[bytes]] = None,
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Entry point nếu connector đã bóc sẵn subject/body/html/images.

    Dùng cho phase integration API sau này.
    """

    inline_image_bytes = inline_image_bytes or []

    plain_from_html = html_to_text(body_html) if body_html else ""

    merged_body = "\n".join([
        body_text or "",
        plain_from_html or ""
    ])

    clean_body = clean_email_body(
        merged_body,
        config=config
    )

    ocr_result = extract_ocr_text_from_images(
        image_items=[
            {
                "source": "raw_inline_image",
                "content_type": "image/unknown",
                "data": img
            }
            for img in inline_image_bytes
        ],
        config=config
    )

    final_email_text = build_final_email_text(
        subject=subject,
        body_text=clean_body,
        ocr_text=ocr_result.get("ocr_text", "")
    )

    return {
        "subject": subject or "",
        "body_text": clean_body,
        "ocr_text": ocr_result.get("ocr_text", ""),
        "email_text": final_email_text,
        "diagnostics": {
            "input_mode": "raw_fields",
            "inline_images_found": len(inline_image_bytes),
            "ocr": ocr_result.get("diagnostics", {})
        }
    }


# ========================
# MIME PARSING
# ========================

def parse_eml_file(eml_path: str) -> EmailMessage:
    """
    Parse file .eml bằng email.policy.default để xử lý MIME tốt hơn.
    """

    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    return msg


def process_email_message(
    msg: EmailMessage,
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Process một EmailMessage:
    - extract subject
    - extract text/plain + text/html
    - extract inline images trong body
    - OCR inline images
    - clean body
    - build email_text cho resolver
    """

    subject = extract_subject(msg)

    body_result = extract_body_texts(msg)
    raw_body_text = body_result.get("body_text", "")
    raw_body_html = body_result.get("body_html", "")

    text_from_html = html_to_text(raw_body_html) if raw_body_html else ""

    merged_body = "\n".join([
        raw_body_text or "",
        text_from_html or ""
    ])

    clean_body = clean_email_body(
        merged_body,
        config=config
    )

    inline_images = extract_inline_images_from_message(
        msg=msg,
        body_html=raw_body_html,
        config=config
    )

    ocr_result = extract_ocr_text_from_images(
        image_items=inline_images,
        config=config
    )

    final_email_text = build_final_email_text(
        subject=subject,
        body_text=clean_body,
        ocr_text=ocr_result.get("ocr_text", "")
    )

    return {
        "subject": subject,
        "body_text": clean_body,
        "ocr_text": ocr_result.get("ocr_text", ""),
        "email_text": final_email_text,
        "diagnostics": {
            "input_mode": "eml",
            "has_text_plain": bool(raw_body_text.strip()),
            "has_text_html": bool(raw_body_html.strip()),
            "inline_images_found": len(inline_images),
            "ocr": ocr_result.get("diagnostics", {})
        }
    }


def extract_subject(msg: EmailMessage) -> str:
    subject = msg.get("subject", "")

    if subject is None:
        return ""

    return str(subject).strip()


def extract_body_texts(msg: EmailMessage) -> Dict[str, str]:
    """
    Extract text/plain và text/html.

    Không lấy attachment text rời ở phase này.
    """

    body_texts = []
    body_htmls = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()

            # Skip attachment
            if disposition == "attachment":
                continue

            if content_type == "text/plain":
                try:
                    body_texts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_texts.append(safe_decode_bytes(payload))

            elif content_type == "text/html":
                try:
                    body_htmls.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_htmls.append(safe_decode_bytes(payload))

    else:
        content_type = msg.get_content_type()

        if content_type == "text/plain":
            try:
                body_texts.append(msg.get_content())
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_texts.append(safe_decode_bytes(payload))

        elif content_type == "text/html":
            try:
                body_htmls.append(msg.get_content())
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_htmls.append(safe_decode_bytes(payload))

    return {
        "body_text": "\n".join(body_texts),
        "body_html": "\n".join(body_htmls)
    }


# ========================
# INLINE IMAGE EXTRACTION
# ========================

def extract_inline_images_from_message(
    msg: EmailMessage,
    body_html: str,
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> List[Dict[str, Any]]:
    """
    Extract ảnh inline/pasted trong body email.

    Hỗ trợ:
    1. MIME image part có Content-ID / disposition inline / no disposition
    2. data:image/...;base64 trong HTML
    """

    images = []

    referenced_cids = extract_cid_references_from_html(body_html)

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()
            content_id = normalize_content_id(part.get("Content-ID", ""))

            if not content_type.startswith("image/"):
                continue

            is_inline = is_inline_image_part(
                disposition=disposition,
                content_id=content_id,
                referenced_cids=referenced_cids
            )

            if config.inline_images_only and not is_inline:
                continue

            data = part.get_payload(decode=True)

            if not data:
                continue

            images.append({
                "source": "mime_inline_image",
                "content_type": content_type,
                "content_id": content_id,
                "disposition": disposition,
                "data": data
            })

    # Extract base64 inline images from HTML
    html_images = extract_base64_images_from_html(body_html)

    images.extend(html_images)

    return images


def extract_cid_references_from_html(body_html: str) -> set:
    """
    Tìm src="cid:xxx" trong HTML.
    """

    if not body_html:
        return set()

    refs = set()

    pattern = r'src=["\']cid:([^"\']+)["\']'

    for match in re.finditer(pattern, body_html, flags=re.IGNORECASE):
        refs.add(normalize_content_id(match.group(1)))

    return refs


def normalize_content_id(content_id: str) -> str:
    """
    Content-ID thường có dạng <image001.png@...>
    Chuẩn hóa về image001.png@...
    """

    if not content_id:
        return ""

    content_id = str(content_id).strip()

    if content_id.startswith("<") and content_id.endswith(">"):
        content_id = content_id[1:-1]

    return content_id.strip()


def is_inline_image_part(
    disposition: Optional[str],
    content_id: str,
    referenced_cids: set
) -> bool:
    """
    Xác định image part có khả năng là ảnh paste trong body không.
    """

    if disposition == "inline":
        return True

    if content_id and content_id in referenced_cids:
        return True

    # Nhiều Outlook inline image không set disposition rõ.
    if content_id and disposition is None:
        return True

    return False


def extract_base64_images_from_html(body_html: str) -> List[Dict[str, Any]]:
    """
    Extract data:image/png;base64,... trong HTML.
    Ít gặp hơn CID nhưng vẫn hỗ trợ.
    """

    if not body_html:
        return []

    images = []

    pattern = r'data:image/(png|jpeg|jpg|gif);base64,([^"\']+)'

    for idx, match in enumerate(re.finditer(pattern, body_html, flags=re.IGNORECASE)):
        ext = match.group(1).lower()
        b64_data = match.group(2)

        try:
            data = base64.b64decode(b64_data)
        except Exception:
            continue

        content_type = f"image/{ext}"

        if ext == "jpg":
            content_type = "image/jpeg"

        images.append({
            "source": "html_base64_image",
            "content_type": content_type,
            "content_id": f"html_base64_{idx}",
            "disposition": "inline",
            "data": data
        })

    return images


# ========================
# OCR
# ========================

def extract_ocr_text_from_images(
    image_items: List[Dict[str, Any]],
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    OCR tất cả inline images hợp lệ.

    Có filter:
    - skip ảnh quá nhỏ
    - skip OCR text không có signal nếu keep_low_signal_ocr=False
    """

    diagnostics = {
        "enabled": config.enable_ocr,
        "images_received": len(image_items),
        "images_ocr_attempted": 0,
        "images_skipped_small": 0,
        "images_skipped_low_signal": 0,
        "ocr_errors": [],
        "ocr_text_items": []
    }

    if not config.enable_ocr:
        return {
            "ocr_text": "",
            "diagnostics": diagnostics
        }

    collected_texts = []

    for idx, item in enumerate(image_items):
        img_bytes = item.get("data")

        if not img_bytes:
            continue

        ocr_result = ocr_image_bytes(
            img_bytes=img_bytes,
            image_index=idx,
            config=config
        )

        if ocr_result.get("skipped_reason") == "small_image":
            diagnostics["images_skipped_small"] += 1
            continue

        if ocr_result.get("error"):
            diagnostics["ocr_errors"].append({
                "image_index": idx,
                "error": ocr_result.get("error")
            })
            continue

        diagnostics["images_ocr_attempted"] += 1

        text = ocr_result.get("text", "").strip()

        if not text:
            continue

        useful = is_probably_useful_ocr_text(text)

        diagnostics["ocr_text_items"].append({
            "image_index": idx,
            "source": item.get("source"),
            "content_type": item.get("content_type"),
            "width": ocr_result.get("width"),
            "height": ocr_result.get("height"),
            "useful": useful,
            "text_preview": text[:300]
        })

        if useful or config.keep_low_signal_ocr:
            collected_texts.append(text)
        else:
            diagnostics["images_skipped_low_signal"] += 1

    final_ocr_text = "\n".join(collected_texts).strip()

    if len(final_ocr_text) > config.max_ocr_text_length:
        final_ocr_text = final_ocr_text[:config.max_ocr_text_length].strip()

    return {
        "ocr_text": final_ocr_text,
        "diagnostics": diagnostics
    }


def ocr_image_bytes(
    img_bytes: bytes,
    image_index: int,
    config: EmailProcessorConfig
) -> Dict[str, Any]:
    """
    OCR một image bytes.

    Yêu cầu môi trường:
    - pillow
    - pytesseract
    - Tesseract OCR binary
    """

    try:
        from PIL import Image
    except Exception as e:
        return {
            "error": f"Pillow not available: {str(e)}"
        }

    try:
        image = Image.open(io.BytesIO(img_bytes))
        width, height = image.size

        area = width * height

        if (
            width < config.min_image_width
            or height < config.min_image_height
            or area < config.min_image_area
        ):
            return {
                "skipped_reason": "small_image",
                "width": width,
                "height": height
            }

        image = image.convert("RGB")

        try:
            import pytesseract
        except Exception as e:
            return {
                "error": f"pytesseract not available: {str(e)}",
                "width": width,
                "height": height
            }

        # Ưu tiên eng+vie, fallback eng nếu máy chưa có vie traineddata.
        try:
            text = pytesseract.image_to_string(
                image,
                lang=config.ocr_lang,
                config="--psm 6"
            )
        except Exception:
            text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--psm 6"
            )

        return {
            "text": text or "",
            "width": width,
            "height": height
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def is_probably_useful_ocr_text(text: str) -> bool:
    """
    Filter OCR noise.

    Mục tiêu:
    - giữ text có khả năng là lỗi / message hệ thống
    - loại logo / banner / chữ ký
    """

    if not text:
        return False

    clean = normalize_for_filter(text)

    if len(clean) < 8:
        return False

    useful_keywords = [
        # English error/system terms
        "error",
        "failed",
        "failure",
        "invalid",
        "exception",
        "warning",
        "cannot",
        "unable",
        "denied",
        "timeout",
        "expired",
        "certificate",
        "autodiscover",
        "outlook",
        "exchange",
        "mailbox",
        "domain",
        "server",
        "service",

        # Vietnamese error terms
        "lỗi",
        "loi",
        "không",
        "khong",
        "không thể",
        "khong the",
        "bị",
        "bi",
        "sai",
        "hết hạn",
        "het han",
        "chứng chỉ",
        "chung chi",
        "đăng nhập",
        "dang nhap",
        "truy cập",
        "truy cap",
        "kết nối",
        "ket noi"
    ]

    for kw in useful_keywords:
        if kw in clean:
            return True

    # Nếu OCR text khá dài thì vẫn giữ, vì có thể là message technical không chứa keyword phổ biến.
    token_count = len(clean.split())

    if token_count >= 12:
        return True

    return False


def normalize_for_filter(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ========================
# BODY CLEANING
# ========================

def clean_email_body(
    text: str,
    config: EmailProcessorConfig = DEFAULT_CONFIG
) -> str:
    """
    Clean body text:
    - normalize whitespace
    - remove email chain
    - remove signature
    - remove boilerplate nhẹ
    """

    text = text or ""

    text = normalize_linebreaks(text)

    if config.remove_chain:
        text = remove_email_chain(text)

    if config.remove_signature:
        text = remove_signature(text)

    text = remove_common_boilerplate(text)

    text = normalize_whitespace(text)

    return text.strip()


def html_to_text(body_html: str) -> str:
    """
    Convert HTML body sang plain text nhẹ.

    Không dùng BeautifulSoup để tránh thêm dependency.
    """

    if not body_html:
        return ""

    text = body_html

    # Remove script/style
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)

    # Replace common breaks/block tags with newline
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?i)</div>", "\n", text)
    text = re.sub(r"(?i)</li>", "\n", text)

    # Remove img tags but keep placeholder to avoid merging text accidentally
    text = re.sub(r"(?is)<img[^>]*>", " [INLINE_IMAGE] ", text)

    # Remove all remaining tags
    text = re.sub(r"(?is)<[^>]+>", " ", text)

    # HTML unescape
    text = html.unescape(text)

    return normalize_whitespace(text)


def normalize_linebreaks(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def remove_email_chain(text: str) -> str:
    """
    Cắt phần reply/forward chain.

    Chỉ lấy nội dung mới nhất ở phía trên.
    """

    if not text:
        return ""

    markers = [
        "-----Original Message-----",
        "----- Forwarded message -----",
        "From:",
        "Sent:",
        "To:",
        "Cc:",
        "Subject:",
        "Original Message",
        "Forwarded message"
    ]

    lower = text.lower()
    cut_positions = []

    for marker in markers:
        pos = lower.find(marker.lower())

        # Chỉ cắt nếu marker xuất hiện sau một đoạn nội dung,
        # tránh cắt nhầm khi body quá ngắn.
        if pos > 30:
            cut_positions.append(pos)

    if cut_positions:
        cut_at = min(cut_positions)
        return text[:cut_at].strip()

    return text.strip()


def remove_signature(text: str) -> str:
    """
    Cắt chữ ký phổ biến.

    Mục tiêu: tránh signature gây nhiễu token scoring.
    """

    if not text:
        return ""

    lines = text.splitlines()

    signature_markers = [
        "trân trọng",
        "tran trong",
        "best regards",
        "kind regards",
        "regards",
        "thanks,",
        "thank you,",
        "sent from my",
        "from my iphone",
        "from my android",
        "phòng",
        "department",
        "tel:",
        "mobile:",
        "email:",
        "website:"
    ]

    cut_line = None

    for idx, line in enumerate(lines):
        clean = line.strip().lower()

        if not clean:
            continue

        # Không cắt quá sớm nếu email chỉ có vài dòng.
        if idx < 2:
            continue

        for marker in signature_markers:
            if marker in clean:
                cut_line = idx
                break

        if cut_line is not None:
            break

    if cut_line is not None:
        return "\n".join(lines[:cut_line]).strip()

    return text.strip()


def remove_common_boilerplate(text: str) -> str:
    """
    Loại bớt các dòng lịch sự quá chung chung.
    Không xóa mạnh để tránh mất nội dung.
    """

    if not text:
        return ""

    noisy_lines = {
        "hi",
        "hello",
        "dear team",
        "hi team",
        "thanks",
        "thank you",
        "nhờ hỗ trợ",
        "nho ho tro",
        "cảm ơn",
        "cam on"
    }

    kept = []

    for line in text.splitlines():
        clean = line.strip()
        lower = clean.lower()

        if lower in noisy_lines:
            continue

        kept.append(line)

    return "\n".join(kept).strip()


# ========================
# FINAL TEXT BUILDER
# ========================

def build_final_email_text(
    subject: str,
    body_text: str,
    ocr_text: str
) -> str:
    """
    Build final email_text đưa vào resolver.

    Có đánh dấu section nhẹ để dễ debug.
    Resolver vẫn chỉ xử lý text bình thường.
    """

    parts = []

    if subject:
        parts.append(f"Subject: {subject.strip()}")

    if body_text:
        parts.append(f"Body:\n{body_text.strip()}")

    if ocr_text:
        parts.append(f"OCR_Text_From_Inline_Image:\n{ocr_text.strip()}")

    return "\n\n".join(parts).strip()


def safe_decode_bytes(data: bytes) -> str:
    """
    Decode bytes an toàn.
    """

    if not data:
        return ""

    for enc in ["utf-8", "utf-16", "cp1258", "cp1252", "latin-1"]:
        try:
            return data.decode(enc)
        except Exception:
            continue

    return data.decode("utf-8", errors="ignore")


# ========================
# DEBUG HELPERS
# ========================

def save_debug_output(result: Dict[str, Any], path: str = "debug_email_processor_output.json"):
    """
    Ghi debug ra file UTF-8 để tránh lỗi console Windows cp1252.
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ========================
# QUICK TEST
# ========================

if __name__ == "__main__":

    # Đổi path này thành file .eml test thực tế.
    test_eml_path = "D:\Temp\Test image.eml"

    if not os.path.exists(test_eml_path):
        result = {
            "error": f"Test file not found: {test_eml_path}",
            "hint": "Export một email có pasted screenshot thành .eml rồi đặt tên sample_email.eml để test."
        }

        save_debug_output(result)
        print("Saved debug_email_processor_output.json")
    else:
        result = process_eml_file(
            eml_path=test_eml_path,
            config=DEFAULT_CONFIG
        )

        save_debug_output(result)
        print("Saved debug_email_processor_output.json")