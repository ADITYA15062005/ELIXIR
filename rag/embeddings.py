"""HuggingFace embedding factory shared by query and ingest paths."""

from __future__ import annotations

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.config import ConfigError, DEFAULT_EMBEDDING_MODEL, Settings, get_settings


def resolve_device(setting: str = "auto") -> str:
    """Resolve TORCH_DEVICE setting to `cpu` or `cuda`."""
    normalized = (setting or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized in {"cpu", "cuda"}:
        if normalized == "cuda" and not torch.cuda.is_available():
            raise ConfigError("TORCH_DEVICE=cuda but CUDA is not available.")
        return normalized
    raise ConfigError(f"Invalid TORCH_DEVICE={setting!r}; use auto, cpu, or cuda.")


def get_embeddings(
    *,
    model_name: str | None = None,
    device: str | None = None,
    settings: Settings | None = None,
) -> HuggingFaceEmbeddings:
    """Build a HuggingFaceEmbeddings instance."""
    if settings is None and (model_name is None or device is None):
        settings = get_settings(require_api_key=False)

    model = model_name or (settings.embedding_model if settings else DEFAULT_EMBEDDING_MODEL)
    device_name = device or resolve_device(
        settings.torch_device_setting if settings else "auto"
    )

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device_name},
    )
