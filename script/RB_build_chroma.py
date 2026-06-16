from app.vector_store import rebuild_runbooks
import re
import json
from pathlib import Path

import numpy as np
import faiss
from docx import Document
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================
"""
Path at company
MODEL_PATH = Path(r"C:\Setup\RB_Build\bge-m3")
RUNBOOK_DIR = Path(r"C:\Setup\RB_Build\Runbook")
OUTPUT_DIR = Path(r"D:\runbook-agent\data")

RUNBOOK_JSON = OUTPUT_DIR / "runbook_data.json"
VECTOR_FILE = OUTPUT_DIR / "runbook_vectors.npy"
FAISS_FILE = OUTPUT_DIR / "runbook_faiss.index"
"""
# path at laptop ================================
MODEL_PATH = Path(r"D:\bge-m3")
RUNBOOK_DIR = Path(r"D:\Runbook")
OUTPUT_DIR = Path(r"D:\LocalAI_v3_refactorCode\data")

RUNBOOK_JSON = OUTPUT_DIR / "runbook_data.json"
VECTOR_FILE = OUTPUT_DIR / "runbook_vectors.npy"
FAISS_FILE = OUTPUT_DIR / "runbook_faiss.index"
#=====================================================


# =====================================================
# UTILS
# =====================================================
def normalize(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_lower(text):
    return normalize(text).lower()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =====================================================
# DETECT SECTION
# =====================================================
def detect_section(text):
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
    Sinh intent tự động từ keyword + description
    Không hard-code theo domain, giới hạn tối đa 5 intent
    """
    candidates = []

    # 1) từ keyword
    if keyword:
        kws = [k.strip() for k in re.split(r"[,;]+", keyword) if k.strip()]
        candidates.extend(kws)

    # 2) bổ sung từ description
    if description:
        desc = normalize_lower(description)

        # tách theo dấu câu + từ nối thông dụng
        parts = re.split(r"(?:,|\.|;| hoặc | và )", desc)

        for p in parts:
            p = normalize(p)
            if 2 <= len(p.split()) <= 8:
                candidates.append(p)

        # enrich nhẹ cho case thực tế hay gặp
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

    # ---- Parse info block
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

    # ---- Parse steps
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

    return {
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


# =====================================================
# BUILD EMBEDDING TEXT
# =====================================================
def build_embedding_text(rb):
    intent_block = "\n".join(f"- {i}" for i in rb.get("intents", []))

    return f"""
Runbook: {rb['title']}
Service: {rb['service']}

Keyword:
{rb['keyword']}

Description:
{rb['description']}

User can ask:
{intent_block}
""".strip()


# =====================================================
# MAIN BUILD
# =====================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model path: {MODEL_PATH}")

    if not RUNBOOK_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy runbook folder: {RUNBOOK_DIR}")

    # =====================================================
    # 1) Parse runbook
    # =====================================================
    runbooks = []

    print("🔄 Parsing runbook DOCX...")
    for f in RUNBOOK_DIR.rglob("*.docx"):
        try:
            rb = parse_docx(f)
            runbooks.append(rb)
            print(f"✅ Parsed: {f.name}")
        except Exception as e:
            print(f"❌ Parse failed: {f.name} -> {e}")

    if not runbooks:
        print("❌ Không tìm thấy file DOCX hợp lệ.")
        return

    # =====================================================
    # 2) Save JSON
    # =====================================================
    save_json(RUNBOOK_JSON, runbooks)
    print(f"✅ Saved: {RUNBOOK_JSON}")

    # =====================================================
    # 3) Build embedding
    # =====================================================
    texts = [build_embedding_text(rb) for rb in runbooks]

    print("🔄 Loading BGE-M3...")
    model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True)

    print("🔄 Encoding vectors...")
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=True
    )
    vectors = np.array(vectors, dtype="float32")

    np.save(VECTOR_FILE, vectors)
    print(f"✅ Saved vectors: {VECTOR_FILE}")

    # =====================================================
    # 4) Build FAISS (optional giữ lại)
    # =====================================================
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(FAISS_FILE))
    print(f"✅ Saved FAISS index: {FAISS_FILE}")

    # =====================================================
    # ✅ 5) Build VECTOR DB (NEW)
    # =====================================================
    print("\n🚀 Building ChromaDB...")
    rebuild_runbooks(runbooks)

    print("\n🎉 BUILD DONE")
    print(f"📘 Total runbooks : {len(runbooks)}")
    print(f"📂 Output folder  : {OUTPUT_DIR}")
