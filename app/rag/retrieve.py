import chromadb
from chromadb.errors import NotFoundError

from app.rag.store import COLLECTION_NAME, DB_PATH, default_embedding_function

_collection = None


class KnowledgeBaseNotReady(RuntimeError):
    pass


def _get_collection():
    """Lazy singleton so we don't reopen the DB on every tool call."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            _collection = client.get_collection(
                name=COLLECTION_NAME, embedding_function=default_embedding_function()
            )
        except NotFoundError as exc:
            raise KnowledgeBaseNotReady(
                "Knowledge base not found. Run `python -m app.rag.ingest` first."
            ) from exc
    return _collection


def search(query: str, n_results: int = 3) -> list[dict]:
    """Return the top-n most relevant KB chunks for a query, with source."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], strict=True
    ):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits
