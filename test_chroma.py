from app.vector_store import add_runbooks, search_runbooks, save_memory, search_memory
import json
from pathlib import Path
from app.vector_store import persist

DATA_FILE = Path("data/runbook_data.json")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    runbooks = json.load(f)


print("🚀 Adding runbooks...")
add_runbooks(runbooks)

print("\n🔍 Search test:")
res = search_runbooks("lỗi mailbox")
print(res)

print("\n💾 Save memory:")
save_memory("lỗi mailbox", "rb_1")

print("\n⚡ Search memory:")
mem = search_memory("lỗi mailbox")
print(mem)