"""Shared settings for the vector store, used by both ingest and retrieve."""

import os

from chromadb.utils import embedding_functions

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Resolve relative paths from the repo root, so the app works no matter which
# folder it is started from (uvicorn, streamlit or python -m).
DB_PATH = os.path.join(REPO_ROOT, os.getenv("CHROMA_DB_PATH", "chroma_store"))
COLLECTION_NAME = "support_kb"


def default_embedding_function():
    """all-MiniLM-L6-v2, run locally through ONNX Runtime.

    Chroma's built-in default uses the same model as sentence-transformers,
    without needing PyTorch, so installs stay small. No API key is needed.
    """
    return embedding_functions.DefaultEmbeddingFunction()
