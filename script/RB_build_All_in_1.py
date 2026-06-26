import re
import json
from pathlib import Path
import hashlib
import requests
from docx import Document


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Folder chứa file runbook DOCX thật
RUNBOOK_DIR = Path(r"D:/runbook")

# Output hiện tại của hệ thống
RUNBOOK_JSON = DATA_DIR / "runbook_data.json"
KEYWORD_CATALOG_JSON = DATA_DIR / "keyword_catalog.json"
KNOWLEDGE_CATALOG_JSON = DATA_DIR / "knowledge_catalog.json"

# Ollama HTTP API
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Nếu True thì bỏ qua cache và gọi lại LLM toàn bộ.
# set bằng True để bỏ qua cache --> force rebuild --> set lại false để sử dụng cache
FORCE_REBUILD_KEYWORD_CATALOG = False

# =====================================================
# UTILS
# =====================================================
def normalize(text):
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_lower(text):
    return normalize(text).lower()

def normalize_keyword_list(raw_keyword):
    """
    Chuẩn hóa keyword từ source runbook thành list.

    Mục tiêu:
    - Source DOCX có thể là string: "Lỗi MFA, reset MFA"
    - Hoặc sau này có thể là list: ["Lỗi MFA", "reset MFA"]
    - Runtime issue_resolver cần list sạch.
    - Không làm mất field keyword string legacy.
    """

    if not raw_keyword:
        return []

    # Case 1: đã là list
    if isinstance(raw_keyword, list):
        raw_items = raw_keyword

    # Case 2: là string từ DOCX/pipeline hiện tại
    elif isinstance(raw_keyword, str):
        # Chỉ split các separator an toàn.
        # KHÔNG split dấu "/" vì có keyword kiểu "gửi/nhận tin nhắn".
        raw_items = re.split(r"[,;\n]+", raw_keyword)

    # Case 3: kiểu khác thì ép về string để không crash pipeline
    else:
        raw_items = [str(raw_keyword)]

    keywords = []
    seen = set()

    for item in raw_items:
        k = normalize(item)

        # bỏ bullet nếu có từ Word
        k = re.sub(r"^[-•\u2022]\s*", "", k)
        k = re.sub(r"^\d+[\.\)]\s*", "", k)
        k = normalize(k)

        if not k:
            continue

        key = normalize_lower(k)

        if key not in seen:
            seen.add(key)
            keywords.append(k)

    return keywords

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_service_name_to_id_map(knowledge_catalog):
    """
    Build mapping:
    service_name (normalize) → service_id
    """

    mapping = {}

    for s in knowledge_catalog:
        name = normalize_lower(s.get("service_name", ""))
        service_id = s.get("service_id")

        if name and service_id:
            mapping[name] = service_id

    return mapping

def split_camel_case(text):
    """
    ActiveDirectory -> Active Directory
    EntraID -> Entra ID
    """

    if not text:
        return ""

    text = re.sub(r"(?<=[a-zà-ỹ])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_service_aliases(service):
    """
    Sinh aliases deterministic từ technical service.
    Không dùng LLM.
    Không phụ thuộc business service catalog.
    """

    service = normalize(service)

    if not service:
        return []

    candidates = []

    # raw
    candidates.append(service)

    # lowercase-readable
    candidates.append(service.lower())

    # camel case split
    split_name = split_camel_case(service)
    if split_name and split_name != service:
        candidates.append(split_name)
        candidates.append(split_name.lower())

    service_lower = service.lower()

    # domain-specific lightweight aliases
    if service_lower in ["ad", "activedirectory", "active directory"]:
        candidates.extend([
            "AD",
            "Active Directory",
            "Domain",
            "Domain Login",
            "Domain Account"
        ])

    if "exchange" in service_lower:
        candidates.extend([
            "Exchange",
            "Microsoft Exchange",
            "Mail Exchange",
            "Mailbox",
            "Outlook",
            "OWA",
            "Email"
        ])

    if "entra" in service_lower:
        candidates.extend([
            "Entra ID",
            "Azure AD",
            "AAD",
            "SSO"
        ])

    if "lync" in service_lower:
        candidates.extend([
            "Lync",
            "Lync Server",
            "Skype for Business"
        ])

    if "iis" == service_lower or "iis" in service_lower:
        candidates.extend([
            "IIS",
            "Web Server",
            "URL Rewrite"
        ])

    if "dhcp" in service_lower:
        candidates.extend([
            "DHCP",
            "DHCP Server",
            "IP cấp phát"
        ])

    if "sql" in service_lower:
        candidates.extend([
            "SQL",
            "SQL Server",
            "Database"
        ])

    if "windows server" in service_lower:
        candidates.extend([
            "Windows Server",
            "Server"
        ])

    # dedupe giữ thứ tự
    output = []
    seen = set()

    for c in candidates:
        c = normalize(c)
        key = normalize_lower(c)

        if not key:
            continue

        if key not in seen:
            seen.add(key)
            output.append(c)

    return output

# =====================================================
# DETECT SECTION
# =====================================================
def detect_section(text):
    """
    Detect section header trong file DOCX.
    """

    t = normalize_lower(text)

    if "thông tin" in t:
        return "info"

    if "điều kiện" in t:
        return "precheck"

    if "các bước" in t:
        return "steps"

    if "kiểm tra" in t:
        return "postcheck"

    return None


# =====================================================
# INTENT GENERATION
# =====================================================
def generate_intents(keyword: str, description: str = "", max_intents=5):
    """
    Sinh intents deterministic từ keyword + description.
    Không dùng LLM.
    Không hard-code domain.

    Mục tiêu:
    - intents hỗ trợ semantic search.
    - intents KHÔNG thay thế keyword gốc.
    """

    candidates = []

    # 1) lấy trực tiếp từ keyword
    if keyword:
        kws = [k.strip() for k in re.split(r"[,;]+", keyword) if k.strip()]
        candidates.extend(kws)

    # 2) bổ sung từ description
    if description:
        desc = normalize_lower(description)

        parts = re.split(r"(?:,|\.|;| hoặc | và )", desc)

        for p in parts:
            p = normalize(p)

            if 2 <= len(p.split()) <= 8:
                candidates.append(p)

        # enrich nhẹ theo logic gốc
        if "mật khẩu" in desc and "hết hạn" in desc:
            candidates.append("mật khẩu hết hạn")

        if "mật khẩu" in desc and "quên" in desc:
            candidates.append("quên mật khẩu")

    # 3) remove duplicate, giữ thứ tự
    seen = set()
    intents = []

    for c in candidates:
        c_norm = normalize_lower(c)

        if not c_norm:
            continue

        if c_norm not in seen:
            seen.add(c_norm)
            intents.append(normalize(c))

    return intents[:max_intents]


# =====================================================
# PARSE DOCX
# =====================================================
def parse_docx(file_path: Path):
    """
    Parse 1 file DOCX thành runbook object.

    Quan trọng:
    - title / service / keyword / description được lấy từ section "Thông tin runbook".
    - keyword là metadata gốc trong runbook.
    - LLM không tham gia bước này.
    """

    doc = Document(str(file_path))

    sections = {
        "info": [],
        "precheck": [],
        "steps": [],
        "postcheck": []
    }

    current_section = None

    for para in doc.paragraphs:
        text = para.text.strip()

        if not text:
            continue

        sec = detect_section(text)

        if sec:
            current_section = sec
            continue

        if current_section:
            sections[current_section].append(text)

    # -------------------------------------------------
    # Parse info block
    # Format kỳ vọng:
    # Tiêu đề: ...
    # Dịch vụ: ...
    # Keyword: ...
    # Kịch bản sử dụng: ...
    # -------------------------------------------------
    info_map = {}

    for line in sections["info"]:
        if ":" in line:
            k, v = line.split(":", 1)
            info_map[normalize_lower(k)] = normalize(v)

    title = (
        info_map.get("tiêu đề")
        or info_map.get("title")
        or file_path.stem
    )

    service = (
        info_map.get("dịch vụ")
        or info_map.get("service")
        or ""
    )

    keyword = (
        info_map.get("keyword")
        or info_map.get("keywords")
        or ""
    )
    keyword_list = normalize_keyword_list(keyword)
    keyword_text = ", ".join(keyword_list)

    description = (
        info_map.get("kịch bản sử dụng")
        or info_map.get("description")
        or ""
    )

    # -------------------------------------------------
    # Parse steps
    # -------------------------------------------------
    def parse_steps(lines):
        steps = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            m = re.match(r"^\d+[\.\)]\s*(.*)", line)

            if m:
                steps.append(normalize(m.group(1)))
            else:
                steps.append(normalize(line))

        return steps

    precheck = [normalize(x) for x in sections["precheck"] if normalize(x)]
    steps = parse_steps(sections["steps"])
    postcheck = [normalize(x) for x in sections["postcheck"] if normalize(x)]

    intents = generate_intents(keyword_text, description, max_intents=5)

    rb = {
        "title": normalize(title),
        "service": normalize(service),
        "keyword": normalize(keyword),
        "keyword_list": keyword_list,
        "description": normalize(description),
        "intents": intents,
        "precheck": precheck,
        "steps": steps,
        "postcheck": postcheck,
        "source_file": str(file_path)
    }

    return rb


# =====================================================
# BUILD RUNBOOK DATA
# =====================================================
def build_all_runbooks():
    """
    Parse toàn bộ DOCX trong RUNBOOK_DIR.
    """

    if not RUNBOOK_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy runbook folder: {RUNBOOK_DIR}")

    runbooks = []

    print("🔄 Parsing runbook DOCX...")

    for f in RUNBOOK_DIR.rglob("*.docx"):
        try:
            rb = parse_docx(f)
            runbooks.append(rb)

            print(
                f"✅ Parsed: {f.name} | "
                f"service={rb.get('service')} | "
                f"keyword={rb.get('keyword')}"
            )

            if not rb.get("keyword"):
                print(f"⚠️ Missing keyword in runbook info block: {f.name}")

            if not rb.get("service"):
                print(f"⚠️ Missing service in runbook info block: {f.name}")

        except Exception as e:
            print(f"❌ Parse failed: {f.name} -> {repr(e)}")

    return runbooks

# =====================================================
# JSON EXTRACTOR
# =====================================================
def extract_json_from_text(text):
    """
    Ollama đôi khi trả kèm text thừa.
    Hàm này cố gắng lấy JSON object đầu tiên.
    """

    if not text:
        return {}

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception:
        return {}
    
# =====================================================
# BUILD KNOWLEDGE CATALOG FOR V4
# =====================================================

def dedupe_text_list(items):
    """
    Dedupe list text, giữ thứ tự.
    """

    output = []
    seen = set()

    for item in items:
        text = normalize(item)
        key = normalize_lower(text)

        if not key:
            continue

        if key not in seen:
            seen.add(key)
            output.append(text)

    return output


def group_runbooks_by_service(runbooks):
    """
    Group runbooks theo technical service gốc từ runbook_data.

    Lưu ý:
    - Đây là service kỹ thuật do người viết runbook điền.
    - Không dùng business service catalog.
    - Không dùng service_mapping của V4_nhỏ.
    """

    grouped = {}

    for rb in runbooks:
        service = normalize(rb.get("service", ""))

        if not service:
            continue

        service_key = normalize_lower(service)

        if service_key not in grouped:
            grouped[service_key] = {
                "service": service,
                "runbooks": []
            }

        grouped[service_key]["runbooks"].append(rb)

    return grouped


def collect_service_keywords_from_runbooks(service_runbooks):
    """
    Collect keywords deterministic cho một technical service.

    Nguồn deterministic:
    - keyword_list
    - keyword string
    - intents
    - title

    Không dùng LLM để sinh keyword chính.
    """

    keywords = []

    for rb in service_runbooks:
        # 1. keyword_list là nguồn tốt nhất nếu đã có
        keyword_list = rb.get("keyword_list") or []

        for kw in keyword_list:
            keywords.append(kw)

        # 2. fallback keyword string
        keyword = rb.get("keyword", "")

        if keyword:
            # normalize_keyword_list đã được bạn thêm trước đó.
            # Nếu file chưa có hàm này thì cần bổ sung ở UTILS.
            keywords.extend(normalize_keyword_list(keyword))

        # 3. intents deterministic từ pipeline
        intents = rb.get("intents", []) or []

        for intent in intents:
            keywords.append(intent)

        # 4. title cũng là signal hữu ích cho service resolver
        title = rb.get("title", "")

        if title:
            keywords.append(title)

    return dedupe_text_list(keywords)


def build_service_runbook_samples(service_runbooks, max_items=10):
    """
    Tạo sample compact để đưa vào prompt LLM.
    Tránh đưa quá nhiều steps/precheck làm prompt dài.
    """

    samples = []

    for rb in service_runbooks[:max_items]:
        samples.append({
            "title": rb.get("title", ""),
            "service": rb.get("service", ""),
            "keyword": rb.get("keyword", ""),
            "keyword_list": rb.get("keyword_list", []),
            "description": rb.get("description", ""),
            "intents": rb.get("intents", [])
        })

    return samples


def build_knowledge_catalog_prompt(service, keywords, service_runbooks):
    """
    Prompt LLM sinh aliases + description cho technical service.

    LLM KHÔNG được:
    - sửa service
    - sinh service mới
    - sinh keyword chính thay deterministic keywords

    LLM CHỈ được:
    - sinh aliases/cách gọi khác cho service
    - sinh description ngắn dựa trên runbook samples
    """

    samples = build_service_runbook_samples(service_runbooks, max_items=3)

    return f"""
Bạn là chuyên gia IT Support trong môi trường ngân hàng.

NHIỆM VỤ:
Tạo aliases và description cho một TECHNICAL SERVICE dựa trên dữ liệu runbook đã có.

QUY TẮC BẮT BUỘC:
- KHÔNG được sửa service gốc.
- KHÔNG được tạo service mới.
- KHÔNG được sinh keyword chính mới.
- aliases chỉ là các cách gọi khác, tên viết tắt, tên phổ biến của service gốc.
- aliases phải liên quan trực tiếp tới dữ liệu runbook được cung cấp.
- description phải ngắn gọn, mô tả service này hỗ trợ nhóm vấn đề gì.
- Chỉ trả về JSON hợp lệ.
- Không giải thích.
- Không markdown.
- Không dùng ```json.

SERVICE GỐC:
{service}

KEYWORDS DETERMINISTIC:
{json.dumps(keywords[:15], ensure_ascii=False, indent=2)}

RUNBOOK SAMPLES:
{json.dumps(samples, ensure_ascii=False, indent=2)}

FORMAT BẮT BUỘC:
{{
  "aliases": ["...", "..."],
  "description": "..."
}}
"""


def normalize_knowledge_catalog_llm_output(data):
    """
    Chuẩn hóa output LLM cho knowledge catalog.
    """

    if not isinstance(data, dict):
        data = {}

    aliases = data.get("aliases") or []
    description = data.get("description") or ""

    if isinstance(aliases, str):
        aliases = [aliases]

    aliases = dedupe_text_list(aliases)

    return {
        "aliases": aliases[:10],
        "description": normalize(description)
    }


def fallback_knowledge_catalog_item(service, keywords):
    """
    Fallback khi LLM lỗi hoặc trả output rỗng.
    Không hard-code domain.
    """

    aliases = dedupe_text_list([service])

    if keywords:
        description = (
            f"Dịch vụ {service}, liên quan đến: "
            + ", ".join(keywords[:8])
            + "."
        )
    else:
        description = f"Dịch vụ {service}."

    return {
        "aliases": aliases,
        "description": description
    }


def extract_aliases_description_llm(service, keywords, service_runbooks):
    """
    Gọi Ollama để sinh aliases + description cho technical service.
    """

    prompt = build_knowledge_catalog_prompt(
        service=service,
        keywords=keywords,
        service_runbooks=service_runbooks
    )

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            print(
                f"❌ KNOWLEDGE LLM HTTP ERROR: {response.status_code} | "
                f"service={service}"
            )
            return fallback_knowledge_catalog_item(service, keywords)

        result = response.json()
        text = result.get("response", "")

        data = extract_json_from_text(text)
        normalized = normalize_knowledge_catalog_llm_output(data)

        # Nếu LLM trả rỗng thì fallback
        if not normalized.get("aliases") and not normalized.get("description"):
            return fallback_knowledge_catalog_item(service, keywords)

        # Luôn thêm service gốc vào aliases để resolver có exact signal
        aliases = dedupe_text_list([service] + normalized.get("aliases", []))

        description = normalized.get("description") or fallback_knowledge_catalog_item(
            service,
            keywords
        )["description"]

        return {
            "aliases": aliases,
            "description": description
        }

    except Exception as e:
        print(
            f"❌ KNOWLEDGE LLM ERROR: service={service} | "
            f"{repr(e)}"
        )
        return fallback_knowledge_catalog_item(service, keywords)


def build_knowledge_catalog_v4(runbooks):
    """
    Build knowledge_catalog.json cho V4.

    Source:
    - runbooks đã parse từ DOCX / runbook_data

    Output schema:
    [
      {
        "service": "...",
        "aliases": [...],
        "keywords": [...],
        "description": "..."
      }
    ]

    Design:
    - service lấy chính xác từ runbook_data
    - keywords lấy deterministic từ runbook_data
    - aliases + description do LLM sinh có kiểm soát
    - KHÔNG dùng business service catalog
    - KHÔNG dùng service_mapping của V4_nhỏ
    """

    grouped = group_runbooks_by_service(runbooks)

    catalog = []

    for _, group in grouped.items():
        service = group["service"]
        service_runbooks = group["runbooks"]

        print(f"🤖 Build V4 knowledge catalog: {service}")

        keywords = collect_service_keywords_from_runbooks(service_runbooks)

        llm_result = extract_aliases_description_llm(
            service=service,
            keywords=keywords,
            service_runbooks=service_runbooks
        )

        item = {
            "service": service,
            "aliases": llm_result.get("aliases", []),
            "keywords": keywords,
            "description": llm_result.get("description", "")
        }

        catalog.append(item)

        print(
            f"✅ Knowledge item generated | "
            f"service={service} | "
            f"aliases={len(item['aliases'])} | "
            f"keywords={len(item['keywords'])}"
        )

    catalog = sorted(
        catalog,
        key=lambda x: normalize_lower(x.get("service", ""))
    )

    print(f"\n📘 V4 KNOWLEDGE CATALOG BUILD SUMMARY")
    print(f"📘 Total services: {len(catalog)}")

    return catalog




# =====================================================
# LLM PROMPT: PATTERNS + SYNONYMS ONLY
# =====================================================
def build_catalog_prompt(runbook):
    """
    LLM chỉ sinh patterns và synonyms.
    Không sinh keyword.
    Không sửa keyword.
    Không sửa service.
    """

    return f"""
Bạn là chuyên gia IT Support tại ngân hàng.

Bạn sẽ nhận được một runbook đã có sẵn:
- title
- service
- keyword
- description
- intents
- steps

QUY TẮC BẮT BUỘC:
- KHÔNG được sinh keyword mới.
- KHÔNG được sửa keyword.
- KHÔNG được sửa service.
- Chỉ tạo patterns và synonyms dựa trên keyword đã cho.
- Không đưa command/script vào patterns hoặc synonyms.
- Chỉ trả về JSON hợp lệ.
- Không giải thích.
- Không markdown.
- Không dùng ```json.

Ý nghĩa:
- keyword là intent gốc do người xây runbook định nghĩa.
- patterns là các câu người dùng nội bộ có thể nói trong thực tế.
- synonyms là các alias/cách gọi khác của keyword.

YÊU CẦU:
- patterns bắt buộc là tiếng Việt.
- patterns phải tự nhiên, gần ngôn ngữ user thật.
- synonyms ưu tiên tiếng Việt.
- synonyms có thể có 1-2 cụm tiếng Anh kỹ thuật phổ biến nếu thật sự cần.
- Không viết kiểu "User request..."
- Tối đa 5 patterns.
- Tối đa 5 synonyms.

FORMAT BẮT BUỘC:

{{
  "patterns": ["...", "..."],
  "synonyms": ["...", "..."]
}}

RUNBOOK INPUT:

Title: {runbook.get("title", "")}
Service: {runbook.get("service", "")}
Keyword: {runbook.get("keyword", "")}
Description: {runbook.get("description", "")}

Intents:
{json.dumps(runbook.get("intents", []), ensure_ascii=False)}

Precheck:
{json.dumps(runbook.get("precheck", []), ensure_ascii=False)[:500]}

Steps:
{json.dumps(runbook.get("steps", []), ensure_ascii=False)[:1000]}

Postcheck:
{json.dumps(runbook.get("postcheck", []), ensure_ascii=False)[:500]}
"""





def normalize_patterns_synonyms(data):
    """
    Chuẩn hóa output từ LLM.
    """

    if not isinstance(data, dict):
        data = {}

    patterns = data.get("patterns") or []
    synonyms = data.get("synonyms") or []

    if isinstance(patterns, str):
        patterns = [patterns]

    if isinstance(synonyms, str):
        synonyms = [synonyms]

    patterns = [normalize(x) for x in patterns if normalize(x)]
    synonyms = [normalize(x) for x in synonyms if normalize(x)]

    # lọc nhẹ output sai kiểu "User ..."
    patterns = [
        p for p in patterns
        if not p.lower().startswith("user ")
    ]

    synonyms = [
        s for s in synonyms
        if not s.lower().startswith("user ")
    ]

    # remove duplicate, giữ thứ tự
    def dedupe(items):
        seen = set()
        output = []

        for item in items:
            key = normalize_lower(item)

            if not key:
                continue

            if key not in seen:
                seen.add(key)
                output.append(item)

        return output

    return {
        "patterns": dedupe(patterns)[:5],
        "synonyms": dedupe(synonyms)[:5]
    }


# =====================================================
# LLM CALL
# =====================================================
def extract_patterns_synonyms_llm(runbook):
    """
    Gọi Ollama HTTP API.

    LLM chỉ trả:
    - patterns
    - synonyms
    """

    prompt = build_catalog_prompt(runbook)

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            print(
                f"❌ LLM HTTP ERROR: {response.status_code} | "
                f"title={runbook.get('title')}"
            )

            return {
                "patterns": [],
                "synonyms": []
            }

        result = response.json()
        text = result.get("response", "")

        data = extract_json_from_text(text)
        normalized = normalize_patterns_synonyms(data)

        return normalized

    except Exception as e:
        print(
            f"❌ LLM ERROR: title={runbook.get('title')} | "
            f"{repr(e)}"
        )

        return {
            "patterns": [],
            "synonyms": []
        }

# =====================================================
# KEYWORD CATALOG CACHE HELPERS
# =====================================================
def make_catalog_cache_key(title, service, keyword):
    """
    Key định danh logical cho một catalog item.

    keyword là dữ liệu gốc từ runbook.
    Nếu title/service/keyword đổi thì coi như item khác.
    """

    return "|".join([
        normalize_lower(title),
        normalize_lower(service),
        normalize_lower(keyword)
    ])


def build_catalog_source_text(runbook):
    """
    Những field ảnh hưởng đến patterns/synonyms.

    Nếu các field này thay đổi thì nên gọi LLM lại.
    """

    payload = {
        "title": runbook.get("title", ""),
        "service": runbook.get("service", ""),
        "keyword": runbook.get("keyword", ""),
        "description": runbook.get("description", ""),
        "intents": runbook.get("intents", []),
        "steps": runbook.get("steps", [])
    }

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def calculate_source_hash(runbook):
    """
    Hash ổn định của nội dung đầu vào cho keyword catalog.
    """

    text = build_catalog_source_text(runbook)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing_keyword_catalog():
    """
    Load keyword_catalog.json cũ để làm persistent cache.

    Return:
        {
            cache_key: catalog_item
        }
    """

    if not KEYWORD_CATALOG_JSON.exists():
        print("ℹ️ No existing keyword catalog cache found.")
        return {}

    try:
        with open(KEYWORD_CATALOG_JSON, "r", encoding="utf-8") as f:
            old_catalog = json.load(f)

        cache = {}

        for item in old_catalog:
            title = item.get("title", "")
            service = item.get("service", "")
            keyword = item.get("keyword", "")

            key = make_catalog_cache_key(title, service, keyword)

            cache[key] = {
                "title": title,
                "service": service,
                "keyword": keyword,
                "keyword_list": item.get("keyword_list") or normalize_keyword_list(keyword),
                "patterns": item.get("patterns", []) or [],
                "synonyms": item.get("synonyms", []) or [],
                "source_hash": item.get("source_hash", "")
            }

        print(f"✅ Loaded keyword catalog cache: {len(cache)} items")

        return cache

    except Exception as e:
        print("⚠️ Failed to load existing keyword catalog cache:", repr(e))
        return {}
# =====================================================
# BUILD KEYWORD CATALOG
# =====================================================
# =====================================================
# BUILD KEYWORD CATALOG WITH CACHE
# =====================================================
def build_keyword_catalog(runbooks, service_mapping):

    """
    Build keyword_catalog.json.

    Design:
    - keyword lấy từ runbook_data.
    - LLM KHÔNG sinh keyword.
    - LLM CHỈ sinh patterns + synonyms.
    - Chỉ gọi LLM khi:
        + runbook mới
        + hoặc title/service/keyword đổi
        + hoặc source_hash đổi
        + hoặc cache cũ thiếu patterns/synonyms
    """

    catalog = []

    if FORCE_REBUILD_KEYWORD_CATALOG:
        print("♻️ FORCE_REBUILD_KEYWORD_CATALOG=True → ignore old cache")
        existing_cache = {}
    else:
        existing_cache = load_existing_keyword_catalog()

    reused_count = 0
    llm_count = 0
    skipped_count = 0
    changed_count = 0

    for rb in runbooks:
        title = rb.get("title", "")
        service = rb.get("service", "")
        keyword = rb.get("keyword", "")
        keyword_list = rb.get("keyword_list") or []
        service_name_norm = normalize_lower(service)
        service_id = service_mapping.get(service_name_norm)
        print(f"🤖 Build catalog: {title}")

        # Keyword là dữ liệu gốc từ runbook.
        # Nếu thiếu keyword thì không cho LLM đoán.
        if not keyword:
            print(f"⚠️ Missing keyword, skip LLM for: {title}")

            catalog.append({
                "title": title,
                "service": service,
                "service_id": service_id,
                "keyword": keyword,
                "keyword_list": keyword_list,
                "patterns": [],
                "synonyms": [],
                "source_hash": calculate_source_hash(rb)
            })

            skipped_count += 1
            continue

        cache_key = make_catalog_cache_key(title, service, keyword)
        current_hash = calculate_source_hash(rb)

        cached_item = existing_cache.get(cache_key)

        if cached_item:
            cached_patterns = cached_item.get("patterns", []) or []
            cached_synonyms = cached_item.get("synonyms", []) or []
            cached_hash = cached_item.get("source_hash", "")

            # Case 1:
            # Cache mới chuẩn, có source_hash và không đổi
            if cached_hash and cached_hash == current_hash and (cached_patterns or cached_synonyms):
                print(
                    f"♻️ Reuse cached catalog | "
                    f"title={title} | keyword={keyword}"
                )

                catalog.append({
                    "title": title,
                    "service": service,
                    "service_id": service_id,
                    "keyword": keyword,
                    "keyword_list": keyword_list,
                    "patterns": cached_patterns,
                    "synonyms": cached_synonyms,
                    "source_hash": current_hash
                })

                reused_count += 1
                continue

            # Case 2:
            # Cache cũ chưa có source_hash.
            # Lần đầu chuyển sang cache version mới, ta vẫn reuse để tránh gọi LLM lại toàn bộ.
            # Sau khi save, item sẽ có source_hash cho các lần sau.
            if not cached_hash and (cached_patterns or cached_synonyms):
                print(
                    f"♻️ Reuse legacy cached catalog and attach hash | "
                    f"title={title} | keyword={keyword}"
                )

                catalog.append({
                    "title": title,
                    "service": service,
                    "service_id": service_id,
                    "keyword": keyword,
                    "keyword_list": keyword_list,
                    "patterns": cached_patterns,
                    "synonyms": cached_synonyms,
                    "source_hash": current_hash
                })

                reused_count += 1
                continue

            # Case 3:
            # Có cache nhưng hash đổi hoặc cache rỗng
            print(
                f"🔄 Catalog changed or empty cache → call LLM | "
                f"title={title} | keyword={keyword}"
            )
            changed_count += 1

        else:
            print(
                f"🆕 New catalog item → call LLM | "
                f"title={title} | keyword={keyword}"
            )

        # Không có cache hợp lệ → gọi LLM
        llm_result = extract_patterns_synonyms_llm(rb)

        item = {
            "title": title,
            "service": service,
            "service_id": service_id,
            "keyword": keyword,
            "keyword_list": keyword_list,
            "patterns": llm_result.get("patterns", []),
            "synonyms": llm_result.get("synonyms", []),
            "source_hash": current_hash
        }

        catalog.append(item)
        llm_count += 1

        print(
            f"✅ Catalog generated | "
            f"title={title} | service={service} | keyword={keyword}"
        )

    print("\n📊 KEYWORD CATALOG BUILD SUMMARY")
    print(f"♻️ Reused from cache : {reused_count}")
    print(f"🤖 Generated by LLM  : {llm_count}")
    print(f"🔄 Changed detected  : {changed_count}")
    print(f"⚠️ Skipped no keyword: {skipped_count}")
    print(f"📘 Total catalog     : {len(catalog)}")

    return catalog

# =====================================================
# MAIN
# =====================================================
def main():
    print("🚀 START BUILD PIPELINE")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1. BUILD RUNBOOK DATA
    # =====================================================
    runbooks = build_all_runbooks()

    if not runbooks:
        print("❌ Không tìm thấy file DOCX hợp lệ.")
        return

    save_json(RUNBOOK_JSON, runbooks)
    print(f"✅ Saved runbook data → {RUNBOOK_JSON}")

    # =====================================================
    # 2. BUILD KNOWLEDGE CATALOG (V4 - TECHNICAL)
    # =====================================================
    knowledge_catalog_v4 = build_knowledge_catalog_v4(runbooks)

    save_json(KNOWLEDGE_CATALOG_JSON, knowledge_catalog_v4)
    print(f"✅ Saved V4 knowledge catalog → {KNOWLEDGE_CATALOG_JSON}")

    # =====================================================
    # 3. BUILD KEYWORD CATALOG (V4 - TECHNICAL)
    # =====================================================
    keyword_catalog = build_keyword_catalog(runbooks, service_mapping=None)
    # ⚠️ nếu build_keyword_catalog của bạn KHÔNG cần service_mapping thì bỏ param này

    save_json(KEYWORD_CATALOG_JSON, keyword_catalog)
    print(f"✅ Saved keyword catalog → {KEYWORD_CATALOG_JSON}")

    print("\n🎉 BUILD PIPELINE DONE")
    print(f"📘 Total runbooks      : {len(runbooks)}")
    print(f"📂 Runbook data        : {RUNBOOK_JSON}")
    print(f"📂 Knowledge catalog   : {KNOWLEDGE_CATALOG_JSON}")
    print(f"📂 Keyword catalog     : {KEYWORD_CATALOG_JSON}")


if __name__ == "__main__":
    main()