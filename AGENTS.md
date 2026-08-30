# ELIXIR (code)

Wine RAG application code lives here.

## Conventions

- Entry UI: `main.py` (Streamlit)
- Stack: LangChain RetrievalQA, HuggingFace embeddings, FAISS, OpenAI-compatible ChatOpenAI (env-driven)
- Keep UI thin; RAG config explicit; **no hardcoded secrets or absolute paths** — use `.env` / `.env.example`
- Nested instructions apply when working under this folder; also see root `AGENTS.md`

## Documentation

Do not put durable docs here. After code changes, ensure `ELIXIROBSIDIAN/03-Changes/` has a dated note (scribe / obsidian-sync).
