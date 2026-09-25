from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"


def load_documents():
    documents = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        documents.append(
            {
                "id": path.stem,
                "text": text,
            }
        )

    return documents


def build_collection():
    documents = load_documents()

    if len(documents) != 8:
        raise RuntimeError(
            f"Expected exactly 8 documents, found {len(documents)}"
        )

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [item["text"] for item in documents]
    ids = [item["id"] for item in documents]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": item["id"],
                "source": item["id"],
            }
            for item in documents
        ],
    )

    print()
    print("=" * 60)
    print("ZEPTO SUPPORT ASSISTANT - INGESTION COMPLETE")
    print("=" * 60)
    print(f"Documents indexed : {collection.count()}")
    print(f"Collection        : {COLLECTION_NAME}")
    print(f"Vector database   : {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build_collection()
