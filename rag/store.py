"""FAISS index path resolution, load, and manifest helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import BASE_DIR, ConfigError, Settings

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def resolve_index_path(settings: Settings) -> Path:
    """Resolve the active FAISS directory.

    Prefer `INDEX_VERSION` → `indexes/<version>/` when set; otherwise use
    `FAISS_INDEX_PATH` (backward compatible with `faiss_index_alt/faiss_wine`).
    """
    if settings.index_version:
        path = (BASE_DIR / "indexes" / settings.index_version).resolve()
        return path
    return settings.faiss_index_path


def load_manifest(index_path: Path) -> dict[str, Any] | None:
    """Load `manifest.json` from an index directory if present."""
    manifest_path = index_path / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read manifest at %s: %s", manifest_path, exc)
    return None


def get_index_version(index_path: Path, manifest: dict[str, Any] | None = None) -> str:
    """Return version from manifest, else the directory name."""
    if manifest is None:
        manifest = load_manifest(index_path)
    if manifest and isinstance(manifest.get("version"), str) and manifest["version"]:
        return manifest["version"]
    return index_path.name


def warn_embedding_mismatch(
    manifest: dict[str, Any] | None,
    embedding_model: str,
) -> None:
    """Log a warning if manifest embedding model ≠ current EMBEDDING_MODEL (MVP)."""
    if not manifest:
        return
    recorded = manifest.get("embedding_model")
    if recorded and recorded != embedding_model:
        logger.warning(
            "Index embedding_model=%r differs from current EMBEDDING_MODEL=%r; "
            "rebuild the index or align the env var.",
            recorded,
            embedding_model,
        )


def load_faiss(
    index_path: Path,
    embeddings: HuggingFaceEmbeddings,
    *,
    embedding_model: str | None = None,
) -> tuple[FAISS, dict[str, Any] | None, str]:
    """Load a local FAISS index.

    Returns:
        (vectorstore, manifest_or_none, index_version)
    """
    if not index_path.exists():
        raise ConfigError(
            f"FAISS index not found at `{index_path}`. "
            "Set FAISS_INDEX_PATH or INDEX_VERSION, or run ingest."
        )

    manifest = load_manifest(index_path)
    if embedding_model:
        warn_embedding_mismatch(manifest, embedding_model)

    db = FAISS.load_local(
        str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    version = get_index_version(index_path, manifest)
    return db, manifest, version
