# Elixir – Wine Recommendation (RAG)

Retrieval-Augmented Generation for personalized wine suggestions from taste descriptions. Built with LangChain, HuggingFace embeddings, FAISS, FastAPI, and Streamlit. Generation uses an **OpenAI-compatible** chat API (OpenAI by default; Groq/OpenRouter via `OPENAI_BASE_URL`).

## Features

- Taste-based querying over wine reviews
- Shared `rag/` module (embeddings, FAISS, RetrievalQA)
- FastAPI recommend API (`POST /v1/recommend`)
- Vite + React product UI (`frontend/`) calling the API
- Offline CSV → versioned FAISS ingest CLI
- Streamlit demo UI over shared `rag/`
- Config via `.env` — no hardcoded API keys or absolute paths

## Setup

```bash
cd ELIXIR
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

- Set `OPENAI_API_KEY`
- Optionally set `LLM_MODEL`, `FAISS_INDEX_PATH`, `OPENAI_BASE_URL`

### Groq example

```env
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

## Run — Streamlit (demo)

```bash
cd ELIXIR
streamlit run main.py
```

Ensure the FAISS index exists at the path in `FAISS_INDEX_PATH` (relative to `ELIXIR/` by default: `faiss_index_alt/faiss_wine`).

## Run — Recommend API

From `ELIXIR/`:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
```

Use `--reload` while developing so prompt/cleanup changes in `rag/chain.py` apply without a manual restart. Bind `0.0.0.0` so other devices on your Wi‑Fi/hotspot can reach the API.

- `GET /health` → `{ "status", "index_loaded", "index_version" }`
- `POST /v1/recommend` body: `{ "query": "oaky cabernet under $30", "k": 3 }`
- OpenAPI docs: `http://127.0.0.1:8001/docs`
- CORS allows Vite origins `http://127.0.0.1:5173` and `http://localhost:5173`

Example:

```bash
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8001/v1/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"crisp mineral white\", \"k\": 3}"
```

## Run — Product UI (Vite + React)

Two processes: API + frontend.

**Terminal 1 — API** (from `ELIXIR/`):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
```

Use `--reload` while developing so prompt/cleanup changes in `rag/chain.py` apply without a manual restart. Bind `0.0.0.0` so other devices on your Wi‑Fi/hotspot can reach the API.

**Terminal 2 — UI** (from `ELIXIR/frontend/`):

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173` on this PC.

**Other devices (phone / tablet on same Wi‑Fi or hotspot):** open `http://<your-pc-lan-ip>:5173` (Vite is bound to all interfaces; it proxies API calls to the local FastAPI process). Find your IP with `ipconfig` (IPv4). Allow Python/Node through Windows Firewall if the page does not load.

Optional: set `VITE_API_BASE=http://<your-pc-lan-ip>:8001` to call the API directly (LAN CORS regex is enabled).

### Stitch design assets

Reference exports live under `frontend/stitch/` (see `frontend/stitch/README.md`). To fetch from Google Stitch when you have a key:

```bash
# set STITCH_API_KEY in the shell (do not commit)
cd ELIXIR/frontend
npm install -D @google/stitch-sdk
npm run fetch:stitch
```

Without a key, drop manual exports into `frontend/stitch/manual-export/`.

## Offline ingest (CSV → FAISS)

Query path never rebuilds embeddings. Build indexes with the CLI.

### Primary dataset: `combined_csv/`

Workspace folder `combined_csv/` (next to `ELIXIR/`) holds Living Liquidz-style catalog CSVs (wine, spirits, beer, etc. — ~1.8k products). Schema uses `Product Title`, `BRAND`, `TYPES`, … (no review `description` column). Ingest synthesizes embeddable text from those fields.

```bash
cd ELIXIR
python -m ingest.cli --csv-dir ../combined_csv --content-mode enriched --out indexes/combined-v1
```

Then point the app at the new index:

```env
FAISS_INDEX_PATH=indexes/combined-v1
```

### Single file / fixtures

```bash
python -m ingest.cli --csv fixtures/sample_catalog.csv --out indexes/sample-catalog
python -m ingest.cli --csv fixtures/sample_wines.csv
```

- Writes LangChain FAISS files + `manifest.json` under `indexes/<version>/`
- `--schema auto|catalog|winemag` (default auto-detect)
- `--content-mode description|enriched` (default from `INGEST_CONTENT_MODE`)
- Winemag column overrides: `--description-col`, `--title-col`, …
- Does **not** copy `combined_csv/` into the repo; does **not** overwrite `faiss_index_alt/`

## Stack

| Piece | Choice |
|-------|--------|
| Product UI | Vite + React (`frontend/` → HTTP API) |
| Demo UI | Streamlit (`main.py` → shared `rag/`) |
| API | FastAPI + Uvicorn (`api/`) |
| Ingest | CLI (`python -m ingest.cli`) |
| Embeddings | HuggingFace (env: `EMBEDDING_MODEL`) |
| Vector store | FAISS (env: `FAISS_INDEX_PATH` / `INDEX_VERSION`) |
| LLM | `langchain_openai.ChatOpenAI` (env-driven) |
| Secrets | `ELIXIR/.env` (gitignored) |

## Dataset

- **Catalog (primary):** `../combined_csv/*.csv` — product catalog; ingest with `--csv-dir`
- **Fixtures:** `fixtures/sample_catalog.csv`, `fixtures/sample_wines.csv` (winemag-style) for smoke builds
- **Legacy demo index:** `faiss_index_alt/faiss_wine` (optional; preserve as-is)
