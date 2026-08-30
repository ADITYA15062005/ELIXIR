"""Curated food-pairing suggestions (positive, brief)."""

from __future__ import annotations

import re

# Kept for older imports; pairing lines no longer append a disclaimer.
PAIRING_DISCLAIMER = ""

# Default body by canonical intent slug.
_DEFAULT_PAIRINGS: dict[str, str] = {
    "beer": "spicy snacks, grilled meats, or salty fries",
    "wine": "cheese, light pasta, or roasted vegetables",
    "whiskey": "dark chocolate, smoked nuts, or grilled meats",
    "tequila": "citrus, ceviche, or spicy Mexican-style dishes",
    "gin": "cucumber salads, seafood, or herbal appetizers",
    "vodka": "smoked fish, pickles, or light canapés",
    "rum": "tropical fruit, barbecue, or chocolate desserts",
    "brandy": "apple desserts, nuts, or mild cheese",
    "sake": "sushi, sashimi, or lightly salted dishes",
    "seltzer": "light salads, picnic snacks, or spicy appetizers",
    "cocktails": "matching the drink's sweetness — salty snacks or citrus bites",
    "mixers": "use with a matching spirit; keep food light and simple",
    "liqueur": "desserts, coffee, or after-dinner cheese",
}

# Optional style overlays: (intent, query cue regex) → more specific body.
_STYLE_OVERLAYS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "beer",
        re.compile(r"\bipas?\b", re.I),
        "spicy food, burgers, or sharp cheddar",
    ),
    (
        "beer",
        re.compile(r"\bstouts?\b", re.I),
        "chocolate desserts, oysters, or roasted meats",
    ),
    (
        "wine",
        re.compile(r"\bred\b", re.I),
        "red meats, aged cheese, or mushroom dishes",
    ),
    (
        "wine",
        re.compile(r"\bwhite\b", re.I),
        "seafood, poultry, or light salads",
    ),
    (
        "whiskey",
        re.compile(r"\bsingle\s+malts?\b", re.I),
        "dark chocolate, smoked cheese, or dried fruit",
    ),
]


def pairing_body(intent: str | None, query: str | None = None) -> str | None:
    """Return the pairing body for an intent, with optional style overlay."""
    if not intent or intent not in _DEFAULT_PAIRINGS:
        return None
    if query:
        for overlay_intent, pattern, body in _STYLE_OVERLAYS:
            if overlay_intent == intent and pattern.search(query):
                return body
    return _DEFAULT_PAIRINGS[intent]


def format_pairing_line(intent: str | None, query: str | None = None) -> str | None:
    """Full post-cleaner line, or None when intent has no curated pairing."""
    body = pairing_body(intent, query)
    if not body:
        return None
    return f"Pairs beautifully with {body}."
