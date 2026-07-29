import os
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_store")
COLLECTION_NAME = "support_kb"

_client = None
_collection = None


def _get_collection():
    """Lazy singleton so we don't reopen the DB on every tool call."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )
    return _collection


def search(query: str, n_results: int = 3) -> list[dict]:
    """Return the top-n most relevant KB chunks for a query, with source."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits
