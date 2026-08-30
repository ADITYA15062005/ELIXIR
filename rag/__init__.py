"""Shared RAG configuration, embeddings, FAISS store, and RetrievalQA chain."""

from rag.chain import build_rag_chain, create_rag_resources, run_recommend
from rag.config import Settings, get_settings, load_env

__all__ = [
    "Settings",
    "build_rag_chain",
    "create_rag_resources",
    "get_settings",
    "load_env",
    "run_recommend",
]
