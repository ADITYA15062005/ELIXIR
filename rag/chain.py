"""RetrievalQA chain and beverage specialist prompt (shared by API + Streamlit)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from langchain.chains import RetrievalQA
except ImportError:  # langchain >= 1.0 moved legacy chains
    from langchain_classic.chains import RetrievalQA

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from rag.config import Settings, get_settings
from rag.embeddings import get_embeddings, resolve_device
from rag.format_card import build_recommend_card
from rag.intent import CategoryFilter, category_predicate, resolve_category
from rag.store import load_faiss, resolve_index_path
from rag.validate import answer_matches_intent, empty_catalog_message

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

# Retrieval broaden / filter knobs (query-time only; no index rebuild).
_MIN_HITS = 1
_FETCH_K_MULT = 5
_FETCH_K_FLOOR = 40
_BROADEN_FLOOR = 200  # unfiltered neighbor scan when filtered hits are thin
_STRICT_NUDGE = (
    "Recommend only from the listed catalog matches for this drink type. "
    "Do not suggest products from other drink families."
)

# Bump when cleanup / card layout changes — included in API responses for debugging.
CLEANER_VERSION = 6

PROMPT_TEMPLATE = """
You write the "why it fits" section for a retail beverage recommendation.

Customer request: {question}

Primary product (use only these facts; do not invent prices or inventory):
{context}

Write PLAIN TEXT only: 2 to 4 short sentences, warm and positive.
Explain why this product fits the request using catalog facts (type, grape,
brand, origin, style, category). You may briefly mention typical character of
a named grape or style when it helps (e.g. Verdejo and bright fruit), without
claiming those notes appear on the product listing.

Forbidden:
- Markdown (#, **, tables, bullets)
- Saying tasting notes are missing / not listed / unknown
- Disclaimers about catalog data
- Food pairing (added separately)
- Recommending a different product
- "Enjoy responsibly" or similar closers
"""


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "context"],
        template=PROMPT_TEMPLATE,
    )


_SECTION_HEADERS = re.compile(
    r"(?im)^\s*(?:"
    r"recommendation|"
    r"best match(?:\s+for[^:]*)?|"
    r"your\b.+\b(?:trio|picks|selection)\b|"
    r"why these three\??|"
    r"quick tasting note|"
    r"tasting note|"
    r"bottom line|"
    r"pairing(?:\s*&\s*service)?(?:\s*tips)?|"
    r"service tips|"
    r"quick summary|"
    r"appearance|"
    r"aroma|"
    r"palate|"
    r"pairing|"
    r"why it'?s a fit|"
    r"why it'?s.?strong.|"
    r"key features"
    r")\s*:?\s*$"
)

_LABEL_PREFIX = re.compile(
    r"(?im)^\s*(?:recommendation|appearance|aroma|palate|pairing|key features|"
    r"why it'?s a fit|service temperature|ideal ageing|body|grapes|origin|price|"
    r"type|volume)\s*:\s*"
)


def _flatten_markdown_table_row(line: str) -> str | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if not (stripped.startswith("|") or stripped.count("|") >= 2):
        return None
    if re.match(r"^[\s|:\-]+$", stripped):
        return ""
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    cells = [c for c in cells if c and not re.fullmatch(r":?-{3,}:?", c)]
    cells = [c for c in cells if c not in {"#", "—", "-"}]
    if not cells:
        return ""
    joined_lower = " ".join(cells).lower()
    header_tokens = ("wine", "origin", "price", "grape", "pairing", "body", "why")
    if "wine" in joined_lower and sum(1 for t in header_tokens if t in joined_lower) >= 2:
        return ""
    return " — ".join(cells)


def to_plain_professional_text(text: str) -> str:
    """Force plain professional prose — Markdown markers cannot survive this pass."""
    cleaned = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\u00a0", " ")
        .replace("\u2011", "-")
        .replace("\u202f", " ")
        .strip()
    )
    cleaned = re.sub(r"(?m)^\s*>\s?", "", cleaned)

    flat_lines: list[str] = []
    for line in cleaned.split("\n"):
        row = _flatten_markdown_table_row(line)
        if row is None:
            flat_lines.append(line)
        elif row:
            flat_lines.append(row)
    cleaned = "\n".join(flat_lines)

    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", cleaned)
    # Nuclear: any leftover markdown markers
    cleaned = cleaned.replace("*", "").replace("#", "").replace("`", "")
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"(?i)<br\s*/?>", " ", cleaned)
    cleaned = re.sub(r"(?i)</?[^>]+>", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-+•]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", "", cleaned)

    cleaned = _SECTION_HEADERS.sub("", cleaned)
    cleaned = _LABEL_PREFIX.sub("", cleaned)
    cleaned = re.sub(r"(?i)\benjoy responsibly!?", "", cleaned)
    cleaned = re.sub(r"(?i)\benjoy your tasting adventure!?", "", cleaned)
    # Drop apologetic / negative catalog lines
    cleaned = re.sub(
        r"(?im)^\s*tasting notes are not listed\.?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\btasting notes are not listed\.?",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\bgeneral suggestion only[^.]*\.?",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\bnot from (the )?catalog listing data\.?",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(sensory|tasting) details (are|were) not (listed|available)\.?",
        "",
        cleaned,
    )
    # Pairing is appended by the system — drop any model-written pairing lines
    cleaned = re.sub(
        r"(?im)^\s*suggested pairing\s*:.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*pairs (beautifully |well )?with\b.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?m)^\s*-{3,}\s*$", "", cleaned)
    cleaned = cleaned.replace("|", " ")

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    if not lines:
        return ""

    product = lines[0].rstrip(":")
    facts: list[str] = []
    rest: list[str] = []
    for ln in lines[1:]:
        if len(ln) < 120 and not ln.endswith(".") and len(facts) < 4 and not rest:
            facts.append(ln.rstrip("."))
        else:
            rest.append(ln)
    paragraphs: list[str] = [product]
    if facts:
        paragraphs.append("; ".join(facts) + ".")
    paragraphs.extend(rest)
    cleaned = "\n\n".join(paragraphs[:3])
    cleaned = cleaned.replace("*", "").replace("#", "").replace("|", " ")
    return cleaned.strip()


def build_llm(settings: Settings) -> ChatOpenAI:
    llm_kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "temperature": 0.1,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**llm_kwargs)


def build_rag_chain(
    db: FAISS,
    llm: ChatOpenAI,
    *,
    k: int,
    prompt: PromptTemplate | None = None,
) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": k}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt or build_prompt()},
    )


def _fetch_k_for(k: int) -> int:
    return max(k * _FETCH_K_MULT, _FETCH_K_FLOOR)


def _biased_search_query(query: str, filt: CategoryFilter) -> str:
    """Append category cue so embeddings land near filtered drink family.

    When API category overrides a semantically distant query (e.g. refreshing +
    tequila), plain similarity neighbors may contain zero category matches
    within fetch_k. Biasing the search text fixes that without an index rebuild.
    """
    cue = (filt.label or filt.intent or filt.exact or "").strip()
    if not cue:
        return query
    q = (query or "").strip()
    if not q:
        return cue
    if cue.lower() in q.lower():
        return q
    return f"{q} {cue}"


def _docs_from_docstore(
    db: FAISS,
    filt: CategoryFilter,
    *,
    limit: int,
) -> list[Document]:
    """Last-resort: scan FAISS docstore for category matches (no similarity)."""
    pred = category_predicate(filt)
    store = getattr(db.docstore, "_dict", None) or {}
    matched: list[Document] = []
    for doc in store.values():
        meta = doc.metadata if hasattr(doc, "metadata") else None
        if pred(meta or {}):
            matched.append(doc)
            if len(matched) >= limit:
                break
    return matched


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        meta = doc.metadata or {}
        cat = meta.get("category") or ""
        title = meta.get("title") or ""
        brand = meta.get("brand") or ""
        header_bits = [b for b in (cat, title, brand) if b]
        header = " | ".join(header_bits)
        body = doc.page_content or ""
        parts.append(f"{header}\n{body}" if header else body)
    return "\n\n".join(parts)


def _post_filter_docs(
    docs: list[Document],
    filt: CategoryFilter,
    *,
    limit: int,
) -> list[Document]:
    pred = category_predicate(filt)
    matched = [d for d in docs if pred(d.metadata or {})]
    return matched[:limit]


def retrieve_docs(
    db: FAISS,
    query: str,
    *,
    k: int,
    filt: CategoryFilter | None = None,
    fetch_k: int | None = None,
) -> list[Document]:
    """Similarity search with optional category metadata filter.

    Prefer FAISS callable filter on a category-biased query; broaden with a
    larger neighbor scan + Python post-filter; finally scan the docstore when
    similarity still yields no category hits but catalog matches exist.
    """
    if filt is None:
        return list(db.similarity_search(query, k=k))

    pred = category_predicate(filt)
    fk = fetch_k if fetch_k is not None else _fetch_k_for(k)
    search_q = _biased_search_query(query, filt)
    try:
        docs = list(
            db.similarity_search(search_q, k=k, filter=pred, fetch_k=fk)
        )
    except TypeError:
        docs = []
    except Exception:
        docs = []

    if len(docs) >= _MIN_HITS:
        return docs[:k]

    # Broaden: larger unfiltered fetch (biased query), then post-filter.
    try:
        broad = list(
            db.similarity_search(search_q, k=max(fk, k * 10, _BROADEN_FLOOR))
        )
    except Exception:
        broad = []
    docs = _post_filter_docs(broad, filt, limit=k)
    if len(docs) >= _MIN_HITS:
        return docs

    # Catalog has matches outside the similarity neighborhood — surface them.
    return _docs_from_docstore(db, filt, limit=k)


def _primary_context(doc: Document) -> str:
    meta = doc.metadata or {}
    bits = [
        f"title={meta.get('title') or ''}",
        f"brand={meta.get('brand') or ''}",
        f"category={meta.get('category') or ''}",
        f"types={meta.get('types') or ''}",
        f"grape={meta.get('variety') or ''}",
        f"grape_type={meta.get('grape_type') or ''}",
        f"style={meta.get('style') or ''}",
        f"country={meta.get('country') or ''}",
        f"region={meta.get('region_1') or meta.get('province') or meta.get('region') or ''}",
        f"volume={meta.get('volume') or ''}",
        f"price={meta.get('price') or ''}",
        f"blurb={(doc.page_content or '')[:400]}",
    ]
    return "\n".join(bits)


def _why_text(
    llm: ChatOpenAI,
    query: str,
    primary: Document,
    *,
    strict: bool = False,
) -> str:
    tmpl = build_prompt()
    question = f"{query}\n\n{_STRICT_NUDGE}" if strict else query
    filled = tmpl.format(question=question, context=_primary_context(primary))
    result = llm.invoke(filled)
    raw = str(result.content) if hasattr(result, "content") else str(result)
    return to_plain_professional_text(raw)


def _generate_answer(
    llm: ChatOpenAI,
    query: str,
    docs: list[Document],
    *,
    prompt: PromptTemplate | None = None,
    strict: bool = False,
) -> str:
    _ = prompt
    if not docs:
        return ""
    return _why_text(llm, query, docs[0], strict=strict)


def run_recommend(
    db: FAISS,
    llm: ChatOpenAI,
    query: str,
    *,
    k: int,
    prompt: PromptTemplate | None = None,
    category: str | None = None,
) -> str:
    """Intent filter → retrieve → structured card (facts + why + pairing)."""
    _ = prompt
    cat_filter = resolve_category(category, query)
    docs = retrieve_docs(db, query, k=max(k, 2), filt=cat_filter)

    if cat_filter is not None and len(docs) < _MIN_HITS:
        return empty_catalog_message(cat_filter)
    if not docs:
        return (
            "We could not find a strong match in the catalog for that request yet. "
            "Try another taste, style, or drink type."
        )

    primary = docs[0]
    alt = docs[1] if len(docs) > 1 else None
    intent = cat_filter.intent if cat_filter else None

    why = _why_text(llm, query, primary, strict=False)
    card = build_recommend_card(
        primary=primary,
        why=why,
        intent=intent,
        query=query,
        alternative=alt,
    )

    if cat_filter is not None and not answer_matches_intent(card, docs, cat_filter):
        retry_docs = retrieve_docs(
            db,
            query,
            k=max(k, 2),
            filt=cat_filter,
            fetch_k=_fetch_k_for(k) * 2,
        )
        if len(retry_docs) < _MIN_HITS:
            return empty_catalog_message(cat_filter)
        primary = retry_docs[0]
        alt = retry_docs[1] if len(retry_docs) > 1 else None
        why = _why_text(llm, query, primary, strict=True)
        card = build_recommend_card(
            primary=primary,
            why=why,
            intent=cat_filter.intent,
            query=query,
            alternative=alt,
        )
        if not answer_matches_intent(card, retry_docs, cat_filter):
            return empty_catalog_message(cat_filter)

    return card


@dataclass
class RagResources:
    settings: Settings
    db: FAISS
    llm: ChatOpenAI
    prompt: PromptTemplate
    index_path: Path
    index_version: str
    device: str
    chain: RetrievalQA


def create_rag_resources(
    settings: Settings | None = None,
    *,
    require_api_key: bool = True,
) -> RagResources:
    """Load embeddings, FAISS, LLM, and a default RetrievalQA chain."""
    settings = settings or get_settings(require_api_key=require_api_key)
    device = resolve_device(settings.torch_device_setting)
    embeddings = get_embeddings(settings=settings, device=device)
    index_path = resolve_index_path(settings)
    db, _manifest, index_version = load_faiss(
        index_path,
        embeddings,
        embedding_model=settings.embedding_model,
    )
    llm = build_llm(settings)
    prompt = build_prompt()
    chain = build_rag_chain(db, llm, k=settings.retriever_k, prompt=prompt)
    return RagResources(
        settings=settings,
        db=db,
        llm=llm,
        prompt=prompt,
        index_path=index_path,
        index_version=index_version,
        device=device,
        chain=chain,
    )
