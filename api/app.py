"""FastAPI app factory — load FAISS once at startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.recommend import router
from rag.chain import create_rag_resources
from rag.config import ConfigError, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.index_loaded = False
    app.state.index_version = None
    app.state.settings = None
    app.state.db = None
    app.state.llm = None
    app.state.prompt = None
    app.state.resources = None

    try:
        settings = get_settings(require_api_key=True)
        resources = create_rag_resources(settings)
        app.state.settings = settings
        app.state.resources = resources
        app.state.db = resources.db
        app.state.llm = resources.llm
        app.state.prompt = resources.prompt
        app.state.index_version = resources.index_version
        app.state.index_loaded = True
        logger.info(
            "Loaded FAISS index version=%s path=%s",
            resources.index_version,
            resources.index_path,
        )
    except ConfigError as exc:
        logger.error("Startup config/index error: %s", exc)
        app.state.index_loaded = False
        app.state.startup_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load RAG resources at startup")
        app.state.index_loaded = False
        app.state.startup_error = str(exc)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Elixir Wine Recommend API",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Local + LAN product UI (Vite). Phones use http://<pc-lan-ip>:5173 via Vite proxy.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
