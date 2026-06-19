import re
import json
from pathlib import Path
import hashlib
import requests
from docx import Document


# =====================================================
# CONFIG
# =====================================================
DATA_DIR = Path("data")

# Folder chứa file runbook DOCX thật
RUNBOOK_DIR = Path(r"D:/runbook")

# Output hiện tại của hệ thống
RUNBOOK_JSON = DATA_DIR / "runbook_data.json"
KEYWORD_CATALOG_JSON = DATA_DIR / "keyword_catalog.json"

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


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    intents = generate_intents(keyword, description, max_intents=5)

    rb = {
        "title": normalize(title),
        "service": normalize(service),
        "keyword": normalize(keyword),
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
def build_keyword_catalog(runbooks):
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

        print(f"🤖 Build catalog: {title}")

        # Keyword là dữ liệu gốc từ runbook.
        # Nếu thiếu keyword thì không cho LLM đoán.
        if not keyword:
            print(f"⚠️ Missing keyword, skip LLM for: {title}")

            catalog.append({
                "title": title,
                "service": service,
                "keyword": keyword,
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
                    "keyword": keyword,
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
                    "keyword": keyword,
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
            "keyword": keyword,
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

    runbooks = build_all_runbooks()

    if not runbooks:
        print("❌ Không tìm thấy file DOCX hợp lệ.")
        return

    # 1) Save source of truth
    save_json(RUNBOOK_JSON, runbooks)
    print(f"✅ Saved runbook data → {RUNBOOK_JSON}")

    # 2) Build derived knowledge for issue_resolver
    keyword_catalog = build_keyword_catalog(runbooks)

    # 3) Save derived catalog
    save_json(KEYWORD_CATALOG_JSON, keyword_catalog)
    print(f"✅ Saved keyword catalog → {KEYWORD_CATALOG_JSON}")

    print("\n🎉 BUILD PIPELINE DONE")
    print(f"📘 Total runbooks      : {len(runbooks)}")
    print(f"📂 Runbook data        : {RUNBOOK_JSON}")
    print(f"📂 Keyword catalog     : {KEYWORD_CATALOG_JSON}")


if __name__ == "__main__":
    main()