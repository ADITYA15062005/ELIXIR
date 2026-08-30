"""Elixir wine RAG — thin Streamlit UI over shared `rag/` module.

Configuration is loaded from ELIXIR/.env (see .env.example). No API keys or
absolute paths are hardcoded.
"""

from __future__ import annotations

import os

import streamlit as st

from rag.chain import create_rag_resources, run_recommend
from rag.config import ConfigError, get_settings

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


def debug_toast(message: str, *, enabled: bool) -> None:
    if enabled:
        st.toast(message)


try:
    settings = get_settings(require_api_key=True)
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

debug_toast("Environment variables loaded.", enabled=settings.app_debug)

try:
    resources = create_rag_resources(settings)
except ConfigError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to initialize RAG: {exc}")
    st.stop()

debug_toast(f"Torch device: {resources.device}", enabled=settings.app_debug)
debug_toast("HuggingFace embeddings initialized.", enabled=settings.app_debug)
debug_toast("FAISS index loaded.", enabled=settings.app_debug)
debug_toast("PromptTemplate initialized.", enabled=settings.app_debug)
debug_toast(f"Chat LLM initialized ({settings.llm_model}).", enabled=settings.app_debug)
debug_toast("RetrievalQA chain initialized.", enabled=settings.app_debug)

st.title("Elixir – Where Taste Meets Technology.")
input_text = st.text_input("Search the topic you want")

if input_text:
    response = run_recommend(
        resources.db,
        resources.llm,
        input_text,
        k=settings.retriever_k,
        prompt=resources.prompt,
    )
    st.write(response)
    debug_toast("Query processed by RAG chain.", enabled=settings.app_debug)
