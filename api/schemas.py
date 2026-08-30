"""Pydantic request/response schemas for the recommend API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Taste / wine preference query")
    k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Retriever top-k; defaults to RETRIEVER_K",
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional drink-type override: intent slug (beer, wine, whiskey, …) "
            "or ingest category (e.g. domestic_beer, tequila)"
        ),
    )


class RecommendResponse(BaseModel):
    answer: str
    query: str
    k: int
    index_version: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    cleaner_version: int = 0


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_version: str | None = None
