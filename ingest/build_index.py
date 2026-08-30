"""Embed Documents and write a versioned FAISS index + manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

from rag.store import MANIFEST_NAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    """Stable hash of multiple files (sorted by basename then content hash)."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name.lower()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def make_version_id(source_sha256: str, *, when: datetime | None = None) -> str:
    """Generate version id: YYYYMMDD-HHMMSS-<short hash>."""
    when = when or datetime.now(timezone.utc)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    short = source_sha256[:7]
    return f"{stamp}-{short}"


def build_manifest(
    *,
    version: str,
    embedding_model: str,
    doc_count: int,
    source_files: list[str],
    source_sha256: str,
    content_mode: str,
    retriever_k: int,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created = created_at or datetime.now(timezone.utc)
    return {
        "version": version,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": embedding_model,
        "doc_count": doc_count,
        "source_files": source_files,
        "source_sha256": source_sha256,
        "content_mode": content_mode,
        "retriever_defaults": {"k": retriever_k},
    }


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return path


def build_and_save_index(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    out_dir: Path,
    *,
    version: str,
    embedding_model: str,
    source_files: list[str],
    source_sha256: str,
    content_mode: str,
    retriever_k: int = 3,
) -> Path:
    """Create FAISS from documents, save_local, write manifest. Returns out_dir."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(out_dir))

    manifest = build_manifest(
        version=version,
        embedding_model=embedding_model,
        doc_count=len(documents),
        source_files=source_files,
        source_sha256=source_sha256,
        content_mode=content_mode,
        retriever_k=retriever_k,
    )
    write_manifest(out_dir, manifest)
    return out_dir
