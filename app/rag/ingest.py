"""
Ingests markdown docs from data/docs/ into a local Chroma vector store.

Run once (or whenever docs change):
    python -m app.rag.ingest
"""

import os
import re

import chromadb
from chromadb.errors import NotFoundError

from app.rag.store import COLLECTION_NAME, DB_PATH, REPO_ROOT, default_embedding_function

DOCS_DIR = os.path.join(REPO_ROOT, "data", "docs")


def chunk_markdown(text: str) -> list[str]:
    """Split a markdown file into chunks on '## ' headers.

    Each chunk keeps its header so retrieved context is self-contained
    and citable back to a specific FAQ entry.
    """
    sections = re.split(r"\n(?=## )", text.strip())
    return [s.strip() for s in sections if s.strip()]


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(docs_dir, filename)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for i, chunk in enumerate(chunk_markdown(content)):
            chunks.append(
                {
                    "id": f"{filename}-{i}",
                    "text": chunk,
                    "source": filename,
                }
            )
    return chunks


def ingest(db_path: str = DB_PATH, docs_dir: str = DOCS_DIR, embedding_function=None) -> int:
    """Rebuild the collection from the docs. Returns the number of chunks stored."""
    client = chromadb.PersistentClient(path=db_path)
    embedding_function = embedding_function or default_embedding_function()

    # Fresh collection each run keeps re-ingestion idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass  # first run: nothing to delete
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )

    docs = load_documents(docs_dir)
    if not docs:
        print(f"No .md files found in {docs_dir}")
        return 0

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )
    print(f"Ingested {len(docs)} chunks into '{COLLECTION_NAME}' at {db_path}")
    return len(docs)


if __name__ == "__main__":
    ingest()
