"""
Ingests markdown docs from data/docs/ into a local Chroma vector store.

Run once (or whenever docs change):
    python -m app.rag.ingest
"""
import os
import re
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "docs")
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_store")
COLLECTION_NAME = "support_kb"


def chunk_markdown(text: str) -> list[str]:
    """Split a markdown file into chunks on '## ' headers.

    Each chunk keeps its header so retrieved context is self-contained
    and citable back to a specific FAQ entry.
    """
    sections = re.split(r"\n(?=## )", text.strip())
    return [s.strip() for s in sections if s.strip()]


def load_documents() -> list[dict]:
    chunks = []
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for i, chunk in enumerate(chunk_markdown(content)):
            chunks.append({
                "id": f"{filename}-{i}",
                "text": chunk,
                "source": filename,
            })
    return chunks


def ingest():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Sentence-transformers runs locally, no API key needed for embeddings.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Fresh collection each run keeps re-ingestion idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    docs = load_documents()
    if not docs:
        print(f"No .md files found in {DOCS_DIR}")
        return

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )
    print(f"Ingested {len(docs)} chunks into '{COLLECTION_NAME}' at {DB_PATH}")


if __name__ == "__main__":
    ingest()
