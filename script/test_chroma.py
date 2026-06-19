import chromadb
from app.load_model import embedding_model


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "runbooks"


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def distance_to_score(dist):
    try:
        return 1 - float(dist)
    except:
        return 0.0


def test_query(query, top_k=4):
    print("\n" + "=" * 60)
    print("🔎 QUERY:", query)

    query_emb = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    docs = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print("🔎 DISTANCES:", distances)

    print("\n📊 RESULTS:")
    for rb, dist in zip(docs, distances):
        score = distance_to_score(dist)

        print(
            f"- {rb.get('title')} | "
            f"service={rb.get('service')} | "
            f"score={score:.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":

    # ✅ test đúng logic dataset
    test_query("remote mailbox lỗi")
    test_query("không nhận được mail 365")
    test_query("lỗi MFA không nhận OTP")
    test_query("quên mật khẩu đăng nhập")
    test_query("tạo user mới")
    test_query("không đăng nhập được")   # edge case