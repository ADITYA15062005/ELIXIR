"""Recommend and health routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import HealthResponse, RecommendRequest, RecommendResponse
from rag.chain import CLEANER_VERSION, run_recommend

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    loaded = bool(getattr(state, "index_loaded", False))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        index_loaded=loaded,
        index_version=getattr(state, "index_version", None),
    )


@router.post("/v1/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
    state = request.app.state
    if not getattr(state, "index_loaded", False):
        raise HTTPException(status_code=503, detail="FAISS index not loaded")

    settings = state.settings
    k = body.k if body.k is not None else settings.retriever_k
    try:
        answer = run_recommend(
            state.db,
            state.llm,
            body.query,
            k=k,
            category=body.category,
        )
    except Exception as exc:  # noqa: BLE001 — surface as 500 for MVP
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecommendResponse(
        answer=answer,
        query=body.query,
        k=k,
        index_version=state.index_version,
        sources=[],
        cleaner_version=CLEANER_VERSION,
    )
