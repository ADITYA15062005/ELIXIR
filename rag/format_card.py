"""Assemble a structured, positive recommend card from catalog docs."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

from rag.pairing import format_pairing_line


def _meta(doc: Document) -> dict[str, Any]:
    return dict(doc.metadata or {})


def _s(meta: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        val = meta.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return None


def _price_line(meta: dict[str, Any]) -> str | None:
    raw = meta.get("price")
    if raw is None or raw == "":
        return None
    try:
        num = float(raw)
        if num.is_integer():
            return f"₹{int(num):,}"
        return f"₹{num:,.1f}"
    except (TypeError, ValueError):
        text = str(raw).strip()
        if not text:
            return None
        if "₹" in text or text.upper().startswith("INR"):
            return text
        return f"₹{text}"


def _type_line(meta: dict[str, Any]) -> str | None:
    types = _s(meta, "types")
    grape = _s(meta, "variety", "grape")
    grape_type = _s(meta, "grape_type")
    style = _s(meta, "style")
    category = _s(meta, "category")
    bits: list[str] = []
    if types:
        bits.append(types)
    if grape and grape_type:
        bits.append(f"{grape} ({grape_type})")
    elif grape:
        bits.append(grape)
    elif grape_type:
        bits.append(grape_type)
    if style and style.upper() not in {b.upper() for b in bits}:
        bits.append(style)
    if not bits and category:
        bits.append(category.replace("_", " "))
    return " · ".join(bits) if bits else None


def _origin_line(meta: dict[str, Any]) -> str | None:
    region = _s(meta, "region_1", "province", "region")
    country = _s(meta, "country")
    if region and country:
        return f"{region}, {country}"
    return region or country


def _title(meta: dict[str, Any], doc: Document) -> str:
    return (
        _s(meta, "title")
        or _s(meta, "brand")
        or (doc.page_content or "").split(".")[0][:80].strip()
        or "Catalog selection"
    )


def _intent_headline(intent: str | None) -> str:
    if not intent:
        return "Recommended"
    labels = {
        "beer": "Recommended Beer",
        "wine": "Recommended Wine",
        "whiskey": "Recommended Whiskey",
        "tequila": "Recommended Tequila",
        "gin": "Recommended Gin",
        "vodka": "Recommended Vodka",
        "rum": "Recommended Rum",
        "brandy": "Recommended Brandy",
        "sake": "Recommended Sake",
        "seltzer": "Recommended Seltzer",
        "cocktails": "Recommended Cocktail",
        "mixers": "Recommended Mixer",
        "liqueur": "Recommended Liqueur",
    }
    return labels.get(intent, f"Recommended {intent.title()}")


def format_product_facts(doc: Document) -> str:
    """Type / Origin / Volume / Price block for one product."""
    meta = _meta(doc)
    lines: list[str] = []
    type_line = _type_line(meta)
    origin = _origin_line(meta)
    volume = _s(meta, "volume")
    price = _price_line(meta)
    if type_line:
        lines.append(f"Type: {type_line}")
    if origin:
        lines.append(f"Origin: {origin}")
    if volume:
        lines.append(f"Volume: {volume}")
    if price:
        lines.append(f"Price: {price}")
    return "\n".join(lines)


def format_alternative(doc: Document) -> str:
    meta = _meta(doc)
    title = _title(meta, doc)
    origin = _origin_line(meta) or _s(meta, "country") or ""
    volume = _s(meta, "volume") or ""
    price = _price_line(meta) or ""
    bits = [b for b in (origin, volume, price) if b]
    detail = ", ".join(bits)
    if detail:
        return f"{title} ({detail})"
    return title


def build_recommend_card(
    *,
    primary: Document,
    why: str,
    intent: str | None,
    query: str,
    alternative: Document | None = None,
) -> str:
    """Full positive card: headline, facts, why, pairing, optional alternative."""
    meta = _meta(primary)
    title = _title(meta, primary)
    headline = _intent_headline(intent)

    sections: list[str] = [f"{headline}: {title}", "", format_product_facts(primary)]

    why_clean = (why or "").strip()
    if why_clean:
        sections.extend(["", "Why this fits your request", why_clean])

    pairing = format_pairing_line(intent, query)
    if pairing:
        sections.extend(["", pairing])

    if alternative is not None:
        sections.extend(
            [
                "",
                "Also worth considering",
                format_alternative(alternative),
            ]
        )

    text = "\n".join(sections)
    # Collapse excess blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()
