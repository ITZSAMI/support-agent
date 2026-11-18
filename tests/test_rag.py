"""Ingest and retrieve against a real Chroma store, with a tiny fake embedding
function so no model has to be downloaded."""

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from app.rag import ingest, retrieve


class KeywordEmbedding(EmbeddingFunction):
    """Counts a few keywords: enough for nearest-neighbour search to work."""

    WORDS = ["return", "damage", "deliver", "address"]

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = [[doc.lower().count(w) + 0.01 for w in self.WORDS] for doc in input]
        # Normalise to length 1, so distance compares topics, not text length.
        return [[x / sum(v * v for v in vec) ** 0.5 for x in vec] for vec in vectors]

    @staticmethod
    def name() -> str:
        return "keyword-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "KeywordEmbedding":
        return KeywordEmbedding()


def test_ingest_twice_then_search(tmp_path, monkeypatch):
    db = str(tmp_path / "store")
    assert ingest.ingest(db_path=db, embedding_function=KeywordEmbedding()) > 0
    count = ingest.ingest(db_path=db, embedding_function=KeywordEmbedding())  # re-run is safe

    monkeypatch.setattr(retrieve, "DB_PATH", db)
    monkeypatch.setattr(retrieve, "default_embedding_function", KeywordEmbedding)
    monkeypatch.setattr(retrieve, "_collection", None)
    hits = retrieve.search("damaged parcel", n_results=1)

    assert count > 0
    assert hits[0]["source"] == "faq.md"
    assert "damage" in hits[0]["text"].lower()


def test_search_before_ingest_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "DB_PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(retrieve, "_collection", None)
    with pytest.raises(retrieve.KnowledgeBaseNotReady, match="app.rag.ingest"):
        retrieve.search("anything")
