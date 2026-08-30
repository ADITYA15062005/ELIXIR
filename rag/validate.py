"""Deterministic recommend validation against drink-type intent."""

from __future__ import annotations

import re
from typing import Any, Sequence

from rag.intent import (
    INTENT_CATEGORY_PATTERNS,
    CategoryFilter,
    category_matches_filter,
)

# Drink-family cues used only to detect clear contradictions in the answer.
_FAMILY_CUES: dict[str, tuple[str, ...]] = {
    "beer": ("beer", "lager", "ale", "stout", "ipa"),
    "wine": ("wine", "rosé", "rose", "sparkling"),
    "whiskey": ("whiskey", "whisky", "scotch", "bourbon", "malt"),
    "tequila": ("tequila", "mezcal"),
    "gin": ("gin",),
    "vodka": ("vodka",),
    "rum": ("rum",),
    "brandy": ("brandy", "cognac"),
    "sake": ("sake",),
    "seltzer": ("seltzer",),
    "cocktails": ("cocktail", "cooler", "rtd"),
    "mixers": ("mixer", "tonic"),
    "liqueur": ("liqueur", "liquer"),
}


def empty_catalog_message(filt: CategoryFilter | str | None) -> str:
    """User-facing message when no matching products are available."""
    if isinstance(filt, CategoryFilter):
        label = filt.label or (filt.intent or "drink")
    elif filt:
        label = str(filt)
    else:
        label = "drink"
    return (
        f"We do not have a matching {label} in the catalog for that request yet. "
        f"Try another style or a broader search."
    )


def docs_match_intent(
    docs: Sequence[Any],
    filt: CategoryFilter | None,
) -> bool:
    """True if at least one retrieved doc's category matches the filter."""
    if filt is None:
        return True
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        if category_matches_filter(meta.get("category"), filt):
            return True
    return False


def _answer_mentions_doc(answer_lower: str, doc: Any) -> bool:
    meta = getattr(doc, "metadata", None) or {}
    title = (meta.get("title") or "").strip().lower()
    brand = (meta.get("brand") or "").strip().lower()
    if title and len(title) >= 3 and title in answer_lower:
        return True
    if brand and len(brand) >= 3 and brand in answer_lower:
        return True
    return False


def _has_family_cue(text_lower: str, intent: str) -> bool:
    for cue in _FAMILY_CUES.get(intent, ()):
        if re.search(rf"\b{re.escape(cue)}\b", text_lower):
            return True
    return False


def answer_matches_intent(
    answer: str,
    docs: Sequence[Any],
    filt: CategoryFilter | None,
) -> bool:
    """Soft deterministic check: filtered docs OK and no clear contradiction.

    Passes when intent is None. Requires ≥1 matching doc. Passes if a matching
    product title/brand appears in the answer, or if there is no strong
    other-family contradiction.
    """
    if filt is None:
        return True
    if not docs_match_intent(docs, filt):
        return False

    matching = [
        d
        for d in docs
        if category_matches_filter(
            (getattr(d, "metadata", None) or {}).get("category"),
            filt,
        )
    ]
    answer_l = (answer or "").lower()
    if any(_answer_mentions_doc(answer_l, d) for d in matching):
        return True

    intent = filt.intent
    if not intent:
        # Exact ingest slug without derived intent — soft pass if docs matched.
        return True

    other_hit = False
    for other, _patterns in INTENT_CATEGORY_PATTERNS.items():
        if other == intent:
            continue
        if _has_family_cue(answer_l, other):
            other_hit = True
            break
    if other_hit and not _has_family_cue(answer_l, intent):
        return False
    return True
