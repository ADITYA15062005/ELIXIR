"""Rule-based drink-type intent + category metadata predicates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Canonical intent → substrings matched against ingest metadata `category`.
INTENT_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "beer": ("beer",),
    "wine": ("wine",),
    "whiskey": ("whiskey", "whisky", "scotch", "malt"),
    "tequila": ("tequila",),
    "gin": ("gin",),
    "vodka": ("vodka",),
    "rum": ("rum",),
    "brandy": ("brandy",),
    "sake": ("sake",),
    "seltzer": ("seltzer",),
    "cocktails": ("cocktail", "cooler"),
    "mixers": ("mixer",),
    "liqueur": ("liquer", "liqueur"),
}

# Human labels for empty-catalog messages.
INTENT_LABELS: dict[str, str] = {
    "beer": "beer",
    "wine": "wine",
    "whiskey": "whiskey",
    "tequila": "tequila",
    "gin": "gin",
    "vodka": "vodka",
    "rum": "rum",
    "brandy": "brandy",
    "sake": "sake",
    "seltzer": "seltzer",
    "cocktails": "cocktail",
    "mixers": "mixer",
    "liqueur": "liqueur",
}

# Synonyms / aliases → canonical intent slug.
_SYNONYM_TO_INTENT: dict[str, str] = {
    "beer": "beer",
    "lager": "beer",
    "ale": "beer",
    "stout": "beer",
    "ipa": "beer",
    "wine": "wine",
    "rosé": "wine",
    "rose": "wine",
    "sparkling": "wine",
    "whiskey": "whiskey",
    "whisky": "whiskey",
    "scotch": "whiskey",
    "bourbon": "whiskey",
    "malt": "whiskey",
    "rye": "whiskey",
    "tequila": "tequila",
    "mezcal": "tequila",
    "gin": "gin",
    "vodka": "vodka",
    "rum": "rum",
    "brandy": "brandy",
    "cognac": "brandy",
    "sake": "sake",
    "seltzer": "seltzer",
    "cocktail": "cocktails",
    "cocktails": "cocktails",
    "cooler": "cocktails",
    "coolers": "cocktails",
    "rtd": "cocktails",
    "mixer": "mixers",
    "mixers": "mixers",
    "liqueur": "liqueur",
    "liquer": "liqueur",
    "liquor": "liqueur",
}

# Query cue patterns → intent (order matters for multi-word first).
_QUERY_CUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsingle\s+malts?\b", re.I), "whiskey"),
    (re.compile(r"\bhard\s+seltzers?\b", re.I), "seltzer"),
    (re.compile(r"\bready[-\s]?to[-\s]?drink\b", re.I), "cocktails"),
    (re.compile(r"\bscotch\b", re.I), "whiskey"),
    (re.compile(r"\bbourbon\b", re.I), "whiskey"),
    (re.compile(r"\bwhisky\b", re.I), "whiskey"),
    (re.compile(r"\bwhiskey\b", re.I), "whiskey"),
    (re.compile(r"\bmalts?\b", re.I), "whiskey"),
    (re.compile(r"\brye\b", re.I), "whiskey"),
    (re.compile(r"\bbeers?\b", re.I), "beer"),
    (re.compile(r"\blagers?\b", re.I), "beer"),
    (re.compile(r"\bales?\b", re.I), "beer"),
    (re.compile(r"\bstouts?\b", re.I), "beer"),
    (re.compile(r"\bipas?\b", re.I), "beer"),
    (re.compile(r"\bwines?\b", re.I), "wine"),
    (re.compile(r"\bros[eé]s?\b", re.I), "wine"),
    (re.compile(r"\bsparkling\b", re.I), "wine"),
    (re.compile(r"\btequilas?\b", re.I), "tequila"),
    (re.compile(r"\bmezcals?\b", re.I), "tequila"),
    (re.compile(r"\bgins?\b", re.I), "gin"),
    (re.compile(r"\bvodkas?\b", re.I), "vodka"),
    (re.compile(r"\brums?\b", re.I), "rum"),
    (re.compile(r"\bbrand(?:y|ies)\b", re.I), "brandy"),
    (re.compile(r"\bcognacs?\b", re.I), "brandy"),
    (re.compile(r"\bsakes?\b", re.I), "sake"),
    (re.compile(r"\bseltzers?\b", re.I), "seltzer"),
    (re.compile(r"\bcocktails?\b", re.I), "cocktails"),
    (re.compile(r"\bcoolers?\b", re.I), "cocktails"),
    (re.compile(r"\brtds?\b", re.I), "cocktails"),
    (re.compile(r"\bmixers?\b", re.I), "mixers"),
    (re.compile(r"\btonics?\b", re.I), "mixers"),
    (re.compile(r"\bliqueurs?\b", re.I), "liqueur"),
    (re.compile(r"\bliquers?\b", re.I), "liqueur"),
]

# Known ingest category slugs (combined_csv / combined-v1).
KNOWN_INGEST_CATEGORIES: frozenset[str] = frozenset(
    {
        "american_whiskey",
        "arrivals",
        "cocktails_coolers",
        "domestic_beer",
        "domestic_wine",
        "imported_brandy",
        "imported_gin",
        "imported_liquer",
        "imported_rum",
        "imported_scotch_blended_whiskey",
        "imported_single_malts",
        "imported_vodka",
        "imported_wine",
        "indian_brandy",
        "indian_gin",
        "indian_liquer",
        "indian_rum",
        "indian_scotch_blended_whiskey",
        "indian_vodka",
        "indian_whiskey",
        "international_beer",
        "irish_whiskey",
        "japanese_whiskey",
        "malts",
        "mixers",
        "rum",
        "sake",
        "seltzer",
        "tequila",
    }
)

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class CategoryFilter:
    """Resolved filter for query-time metadata matching."""

    intent: str | None
    patterns: tuple[str, ...]
    exact: str | None = None
    label: str = "drink"


def intent_from_ingest_category(category: str | None) -> str | None:
    """Map an ingest category slug to a canonical intent, if possible."""
    if not category:
        return None
    cat = category.strip().lower()
    for intent, patterns in INTENT_CATEGORY_PATTERNS.items():
        if _category_matches_patterns(cat, patterns):
            return intent
    return None


def normalize_category(raw: str | None) -> CategoryFilter | None:
    """Normalize API/user category string to a CategoryFilter.

    - Synonym / intent slug → pattern filter for that intent
    - Known ingest slug (e.g. domestic_beer) → exact match + derived intent
    - Unknown bare slug that looks like ingest → exact match
    - Empty / unusable → None
    """
    if raw is None:
        return None
    text = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None

    # Multi-word API values may arrive with spaces already collapsed to _
    spaced = text.replace("_", " ")
    synonym_key = spaced if spaced in _SYNONYM_TO_INTENT else text
    if synonym_key in _SYNONYM_TO_INTENT:
        intent = _SYNONYM_TO_INTENT[synonym_key]
        return CategoryFilter(
            intent=intent,
            patterns=INTENT_CATEGORY_PATTERNS[intent],
            exact=None,
            label=INTENT_LABELS.get(intent, intent),
        )

    if text in INTENT_CATEGORY_PATTERNS:
        return CategoryFilter(
            intent=text,
            patterns=INTENT_CATEGORY_PATTERNS[text],
            exact=None,
            label=INTENT_LABELS.get(text, text),
        )

    if text in KNOWN_INGEST_CATEGORIES or _SLUG_RE.match(text):
        intent = intent_from_ingest_category(text)
        label = INTENT_LABELS.get(intent, text.replace("_", " ")) if intent else text.replace(
            "_", " "
        )
        return CategoryFilter(
            intent=intent,
            patterns=(),
            exact=text,
            label=label,
        )

    return None


def detect_intent(query: str) -> str | None:
    """Detect a single drink-type intent from the query; None if ambiguous."""
    if not query or not query.strip():
        return None
    found: list[str] = []
    for pattern, intent in _QUERY_CUE_PATTERNS:
        if pattern.search(query) and intent not in found:
            found.append(intent)
    if len(found) == 1:
        return found[0]
    return None


def resolve_category(
    api_category: str | None,
    query: str,
) -> CategoryFilter | None:
    """Precedence: explicit API category → query detect → ambiguous (None)."""
    if api_category is not None and str(api_category).strip():
        resolved = normalize_category(api_category)
        if resolved is not None:
            return resolved
        # Unknown API value: still try as exact slug filter
        slug = str(api_category).strip().lower().replace("-", "_").replace(" ", "_")
        if slug:
            intent = intent_from_ingest_category(slug)
            return CategoryFilter(
                intent=intent,
                patterns=(),
                exact=slug,
                label=(INTENT_LABELS.get(intent, slug.replace("_", " ")) if intent else slug.replace("_", " ")),
            )

    intent = detect_intent(query)
    if intent is None:
        return None
    return CategoryFilter(
        intent=intent,
        patterns=INTENT_CATEGORY_PATTERNS[intent],
        exact=None,
        label=INTENT_LABELS.get(intent, intent),
    )


def _category_matches_patterns(category: str, patterns: tuple[str, ...]) -> bool:
    """Substring match on category slug with word-ish boundaries for short tokens."""
    cat = category.strip().lower()
    if not cat:
        return False
    for pat in patterns:
        p = pat.lower()
        # Prefer token/substring on underscore-separated slugs (e.g. rum in indian_rum).
        if f"_{p}_" in f"_{cat}_" or cat == p or cat.startswith(f"{p}_") or cat.endswith(f"_{p}"):
            return True
        if p in cat and len(p) >= 4:
            return True
    return False


def category_matches_filter(metadata_category: str | None, filt: CategoryFilter) -> bool:
    """Return True if a doc's metadata category satisfies the filter."""
    if filt.exact is not None:
        return (metadata_category or "").strip().lower() == filt.exact
    if not filt.patterns:
        return False
    return _category_matches_patterns(metadata_category or "", filt.patterns)


def category_predicate(filt: CategoryFilter) -> Callable[[dict], bool]:
    """Callable suitable for FAISS `filter=` (receives metadata dict)."""

    def _pred(metadata: dict) -> bool:
        return category_matches_filter(
            metadata.get("category") if isinstance(metadata, dict) else None,
            filt,
        )

    return _pred
