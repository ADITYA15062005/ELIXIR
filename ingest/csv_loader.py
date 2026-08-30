"""Load CSV rows into LangChain Documents (stdlib csv only).

Supports:
- winemag-style review CSVs (`description`, …)
- Living Liquidz catalog CSVs from workspace `combined_csv/` (`Product Title`, …)
"""

from __future__ import annotations

import csv
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover — older langchain layouts
    from langchain.schema import Document

from ingest.schema import (
    CatalogColumnMapping,
    ColumnMapping,
    SchemaKind,
    category_from_filename,
    detect_schema_kind,
    optional_str,
    parse_optional_float,
    parse_optional_int,
)


def build_winemag_page_content(
    row: dict[str, str],
    mapping: ColumnMapping,
    *,
    content_mode: str,
) -> str | None:
    """Build document text; return None if description is empty."""
    description = optional_str(row.get(mapping.description))
    if not description:
        return None

    if content_mode != "enriched":
        return description

    title = optional_str(row.get(mapping.title))
    variety = optional_str(row.get(mapping.variety))
    winery = optional_str(row.get(mapping.winery))
    bits: list[str] = []
    if title:
        bits.append(title)
    mid_parts = [p for p in (variety, f"from {winery}" if winery else None) if p]
    if mid_parts:
        bits.append(" ".join(mid_parts))
    if not bits:
        return description
    return f"{'. '.join(bits)}. {description}"


def build_catalog_page_content(
    row: dict[str, str],
    mapping: CatalogColumnMapping,
    *,
    content_mode: str,
    category: str | None,
) -> str | None:
    """Synthesize embeddable text from product attributes (no review description)."""
    title = optional_str(row.get(mapping.title))
    brand = optional_str(row.get(mapping.brand))
    if not title and not brand:
        return None

    types = optional_str(row.get(mapping.types))
    grape = optional_str(row.get(mapping.grape))
    grape_type = optional_str(row.get(mapping.grape_type))
    style = optional_str(row.get(mapping.style))
    flavour = optional_str(row.get(mapping.flavour))
    region = optional_str(row.get(mapping.region))
    country = optional_str(row.get(mapping.country))
    age = optional_str(row.get(mapping.age))
    cask = optional_str(row.get(mapping.cask_type))
    vintage = optional_str(row.get(mapping.vintage))
    volume = optional_str(row.get(mapping.volume))
    price = optional_str(row.get(mapping.price))

    # Compact “description” mode: title + key identity
    parts: list[str] = []
    if title:
        parts.append(title)
    identity = [p for p in (brand, types, category) if p]
    if identity:
        parts.append(" · ".join(identity))

    if content_mode == "enriched":
        detail_bits: list[str] = []
        if style:
            detail_bits.append(f"Style: {style}")
        if flavour:
            detail_bits.append(f"Flavour: {flavour}")
        grape_bits = [p for p in (grape, grape_type) if p]
        if grape_bits:
            detail_bits.append(f"Grape: {' / '.join(grape_bits)}")
        place = ", ".join(p for p in (region, country) if p)
        if place:
            detail_bits.append(f"Origin: {place}")
        if age:
            detail_bits.append(f"Age: {age}")
        if cask:
            detail_bits.append(f"Cask: {cask}")
        if vintage:
            detail_bits.append(f"Vintage: {vintage}")
        if volume:
            detail_bits.append(f"Volume: {volume}")
        if price:
            detail_bits.append(f"Price: {price}")
        parts.extend(detail_bits)
    else:
        # Still include high-signal fields when present
        extras = [p for p in (style, flavour, grape, region, country) if p]
        if extras:
            parts.append(", ".join(extras))

    text = ". ".join(parts).strip()
    return text or None


def winemag_row_to_document(
    row: dict[str, str],
    *,
    mapping: ColumnMapping,
    source_file: str,
    source_row: int,
    ingest_version: str,
    embedding_model: str,
    content_mode: str,
    category: str | None = None,
) -> Document | None:
    page_content = build_winemag_page_content(row, mapping, content_mode=content_mode)
    if page_content is None:
        return None

    id_raw = optional_str(row.get(mapping.id_col))
    row_id = id_raw if id_raw is not None else str(source_row)
    doc_id = f"csv:{source_file}:{row_id}"

    metadata = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_row": source_row,
        "schema": SchemaKind.WINEMAG.value,
        "category": category,
        "title": optional_str(row.get(mapping.title)),
        "variety": optional_str(row.get(mapping.variety)),
        "winery": optional_str(row.get(mapping.winery)),
        "country": optional_str(row.get(mapping.country)),
        "province": optional_str(row.get(mapping.province)),
        "region_1": optional_str(row.get(mapping.region_1)),
        "points": parse_optional_int(row.get(mapping.points)),
        "price": parse_optional_float(row.get(mapping.price)),
        "taster_name": optional_str(row.get(mapping.taster_name)),
        "ingest_version": ingest_version,
        "embedding_model": embedding_model,
    }
    return Document(page_content=page_content, metadata=metadata)


def catalog_row_to_document(
    row: dict[str, str],
    *,
    mapping: CatalogColumnMapping,
    source_file: str,
    source_row: int,
    ingest_version: str,
    embedding_model: str,
    content_mode: str,
    category: str | None = None,
) -> Document | None:
    cat = category or category_from_filename(source_file)
    page_content = build_catalog_page_content(
        row, mapping, content_mode=content_mode, category=cat
    )
    if page_content is None:
        return None

    link = optional_str(row.get(mapping.link))
    row_id = link or str(source_row)
    doc_id = f"csv:{source_file}:{row_id}"

    metadata = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_row": source_row,
        "schema": SchemaKind.CATALOG.value,
        "category": cat,
        "title": optional_str(row.get(mapping.title)),
        "brand": optional_str(row.get(mapping.brand)),
        "types": optional_str(row.get(mapping.types)),
        "variety": optional_str(row.get(mapping.grape)),
        "grape_type": optional_str(row.get(mapping.grape_type)),
        "winery": optional_str(row.get(mapping.brand)),
        "country": optional_str(row.get(mapping.country)),
        "province": optional_str(row.get(mapping.region)),
        "region_1": optional_str(row.get(mapping.region)),
        "style": optional_str(row.get(mapping.style)),
        "flavour": optional_str(row.get(mapping.flavour)),
        "age": optional_str(row.get(mapping.age)),
        "cask_type": optional_str(row.get(mapping.cask_type)),
        "vintage": optional_str(row.get(mapping.vintage)),
        "volume": optional_str(row.get(mapping.volume)),
        "label": optional_str(row.get(mapping.label)),
        "native": optional_str(row.get(mapping.native)),
        "company": optional_str(row.get(mapping.company)),
        "link": link,
        "points": None,
        "price": parse_optional_float(row.get(mapping.price)),
        "ingest_version": ingest_version,
        "embedding_model": embedding_model,
    }
    return Document(page_content=page_content, metadata=metadata)


def load_documents_from_csv(
    csv_path: Path,
    *,
    ingest_version: str,
    embedding_model: str,
    content_mode: str = "description",
    schema_kind: SchemaKind | None = None,
    winemag_mapping: ColumnMapping | None = None,
    catalog_mapping: CatalogColumnMapping | None = None,
    category: str | None = None,
) -> list[Document]:
    """Read CSV → list of Documents; skip rows that cannot form page_content."""
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    source_file = csv_path.name
    documents: list[Document] = []
    winemag_mapping = winemag_mapping or ColumnMapping()
    catalog_mapping = catalog_mapping or CatalogColumnMapping()

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else None
        kind = schema_kind or detect_schema_kind(fieldnames)

        if kind == SchemaKind.CATALOG:
            catalog_mapping.validate_headers(fieldnames)
        else:
            winemag_mapping.validate_headers(fieldnames)

        for i, row in enumerate(reader, start=1):
            normalized = {k: (v if v is not None else "") for k, v in row.items() if k}
            if kind == SchemaKind.CATALOG:
                doc = catalog_row_to_document(
                    normalized,
                    mapping=catalog_mapping,
                    source_file=source_file,
                    source_row=i,
                    ingest_version=ingest_version,
                    embedding_model=embedding_model,
                    content_mode=content_mode,
                    category=category,
                )
            else:
                doc = winemag_row_to_document(
                    normalized,
                    mapping=winemag_mapping,
                    source_file=source_file,
                    source_row=i,
                    ingest_version=ingest_version,
                    embedding_model=embedding_model,
                    content_mode=content_mode,
                    category=category,
                )
            if doc is not None:
                documents.append(doc)

    if not documents:
        raise ValueError(
            f"No documents produced from {source_file} "
            "(all rows missing required text fields or file empty)."
        )
    return documents


def load_documents_from_csv_dir(
    csv_dir: Path,
    *,
    ingest_version: str,
    embedding_model: str,
    content_mode: str = "description",
    schema_kind: SchemaKind | None = None,
    pattern: str = "*.csv",
) -> tuple[list[Document], list[str]]:
    """Load and merge all matching CSVs under a directory.

    Returns (documents, source_file basenames sorted).
    """
    csv_dir = csv_dir.resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    paths = sorted(csv_dir.glob(pattern))
    if not paths:
        raise ValueError(f"No CSV files matching {pattern!r} in {csv_dir}")

    all_docs: list[Document] = []
    source_files: list[str] = []
    for path in paths:
        docs = load_documents_from_csv(
            path,
            ingest_version=ingest_version,
            embedding_model=embedding_model,
            content_mode=content_mode,
            schema_kind=schema_kind,
            category=category_from_filename(path.name),
        )
        all_docs.extend(docs)
        source_files.append(path.name)

    return all_docs, source_files
