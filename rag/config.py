"""Shared configuration loaded from ELIXIR/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FAISS_INDEX_PATH = "faiss_index_alt/faiss_wine"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_RETRIEVER_K = 3
DEFAULT_INGEST_CONTENT_MODE = "description"


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


def load_env() -> None:
    """Load `.env` and normalize LangChain tracing env vars."""
    load_dotenv(BASE_DIR / ".env")

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower()
    os.environ["LANGCHAIN_TRACING_V2"] = (
        "true" if tracing in {"1", "true", "yes"} else "false"
    )
    lc_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if lc_key:
        os.environ["LANGCHAIN_API_KEY"] = lc_key


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable `{name}`. "
            "Copy `.env.example` to `.env` and set it."
        )
    return value


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    llm_model: str
    openai_base_url: str
    embedding_model: str
    faiss_index_path: Path
    index_version: str | None
    retriever_k: int
    torch_device_setting: str
    app_debug: bool
    ingest_content_mode: str
    api_host: str
    api_port: int


def get_settings(*, require_api_key: bool = True) -> Settings:
    """Load settings from the environment.

    Args:
        require_api_key: When False (e.g. ingest-only), skip OPENAI_API_KEY.
    """
    load_env()

    api_key = _require_env("OPENAI_API_KEY") if require_api_key else os.getenv(
        "OPENAI_API_KEY", ""
    ).strip()

    llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    embedding_model = (
        os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )

    index_rel = (
        os.getenv("FAISS_INDEX_PATH", DEFAULT_FAISS_INDEX_PATH).strip()
        or DEFAULT_FAISS_INDEX_PATH
    )
    index_path = Path(index_rel)
    if not index_path.is_absolute():
        index_path = (BASE_DIR / index_path).resolve()

    index_version = os.getenv("INDEX_VERSION", "").strip() or None

    try:
        retriever_k = int(os.getenv("RETRIEVER_K", str(DEFAULT_RETRIEVER_K)).strip() or str(DEFAULT_RETRIEVER_K))
    except ValueError as exc:
        raise ConfigError("RETRIEVER_K must be an integer.") from exc

    content_mode = (
        os.getenv("INGEST_CONTENT_MODE", DEFAULT_INGEST_CONTENT_MODE).strip().lower()
        or DEFAULT_INGEST_CONTENT_MODE
    )
    if content_mode not in {"description", "enriched"}:
        raise ConfigError(
            f"Invalid INGEST_CONTENT_MODE={content_mode!r}; use description or enriched."
        )

    try:
        api_port = int(os.getenv("API_PORT", "8001").strip() or "8001")
    except ValueError as exc:
        raise ConfigError("API_PORT must be an integer.") from exc

    return Settings(
        openai_api_key=api_key,
        llm_model=llm_model,
        openai_base_url=base_url,
        embedding_model=embedding_model,
        faiss_index_path=index_path,
        index_version=index_version,
        retriever_k=retriever_k,
        torch_device_setting=os.getenv("TORCH_DEVICE", "auto").strip().lower() or "auto",
        app_debug=_env_bool("APP_DEBUG"),
        ingest_content_mode=content_mode,
        api_host=os.getenv("API_HOST", "0.0.0.0").strip() or "0.0.0.0",
        api_port=api_port,
    )
