"""Unit checks for drink-type intent, pairing, and validate helpers."""

from __future__ import annotations

from types import SimpleNamespace

from rag.intent import (
    category_matches_filter,
    detect_intent,
    resolve_category,
)
from rag.pairing import format_pairing_line, pairing_body
from rag.validate import (
    answer_matches_intent,
    docs_match_intent,
    empty_catalog_message,
)


def test_detect_beer_and_whiskey() -> None:
    assert detect_intent("crisp lager beer under 200") == "beer"
    assert detect_intent("smooth single malt") == "whiskey"
    assert detect_intent("nice scotch whisky") == "whiskey"


def test_detect_ambiguous() -> None:
    assert detect_intent("nice gift drink") is None
    assert detect_intent("wine or beer") is None
    assert detect_intent("something refreshing") is None


def test_resolve_api_overrides_query() -> None:
    filt = resolve_category("tequila", "something refreshing")
    assert filt is not None
    assert filt.intent == "tequila"
    assert resolve_category(None, "nice gift drink") is None


def test_category_patterns() -> None:
    beer = resolve_category(None, "lager beer")
    assert category_matches_filter("domestic_beer", beer)
    assert not category_matches_filter("imported_wine", beer)
    whiskey = resolve_category(None, "single malt")
    assert category_matches_filter("imported_single_malts", whiskey)


def test_pairing_line_positive() -> None:
    line = format_pairing_line("beer", "crisp lager beer")
    assert line is not None
    assert line.startswith("Pairs beautifully with ")
    assert "not from catalog" not in line.lower()
    assert "general suggestion" not in line.lower()
    assert format_pairing_line(None) is None
    assert pairing_body("beer", "hoppy ipa") == "spicy food, burgers, or sharp cheddar"


def test_empty_catalog_and_validate() -> None:
    filt = resolve_category(None, "beer please")
    assert "We do not have a matching beer" in empty_catalog_message(filt)
    doc_beer = SimpleNamespace(
        metadata={"category": "domestic_beer", "title": "Kingfisher", "brand": "UB"}
    )
    doc_wine = SimpleNamespace(
        metadata={"category": "imported_wine", "title": "Bordeaux", "brand": "X"}
    )
    assert docs_match_intent([doc_beer], filt)
    assert not docs_match_intent([doc_wine], filt)
    assert answer_matches_intent("Try Kingfisher lager.", [doc_beer], filt)
    assert not answer_matches_intent("This fine wine is perfect.", [doc_beer], filt)
