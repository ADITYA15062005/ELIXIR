"""CSV column mapping and validation for wine reviews and catalog CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SchemaKind(str, Enum):
    WINEMAG = "winemag"
    CATALOG = "catalog"


# Living Liquidz / combined_csv headers
CATALOG_TITLE = "Product Title"
CATALOG_PRICE = "Price (₹)"
CATALOG_REQUIRED_ANY = (CATALOG_TITLE, "BRAND")


@dataclass(frozen=True)
class ColumnMapping:
    """Maps CSV headers to document fields (winemag-style defaults)."""

    description: str = "description"
    title: str = "title"
    variety: str = "variety"
    winery: str = "winery"
    province: str = "province"
    region_1: str = "region_1"
    country: str = "country"
    points: str = "points"
    price: str = "price"
    taster_name: str = "taster_name"
    id_col: str = "id"

    def validate_headers(self, fieldnames: list[str] | None) -> None:
        if not fieldnames:
            raise ValueError("CSV has no header row.")
        headers = {h.strip() for h in fieldnames if h}
        if self.description not in headers:
            raise ValueError(
                f"Required description column {self.description!r} not found in CSV. "
                f"Available columns: {sorted(headers)}"
            )


@dataclass(frozen=True)
class CatalogColumnMapping:
    """Maps Living Liquidz / combined_csv headers to document fields."""

    title: str = CATALOG_TITLE
    brand: str = "BRAND"
    types: str = "TYPES"
    grape: str = "GRAPE"
    grape_type: str = "GRAPE TYPE"
    country: str = "COUNTRY"
    region: str = "REGION"
    style: str = "STYLE"
    flavour: str = "FLAVOUR"
    age: str = "AGE"
    bottler: str = "BOTTLER"
    cask_type: str = "CASK TYPE"
    vintage: str = "VINTAGE"
    colouring: str = "COLOURING"
    company: str = "COMPANY"
    price: str = CATALOG_PRICE
    volume: str = "Volume"
    label: str = "LABEL"
    native: str = "NATIVE"
    link: str = "link"

    def validate_headers(self, fieldnames: list[str] | None) -> None:
        if not fieldnames:
            raise ValueError("CSV has no header row.")
        headers = {h.strip() for h in fieldnames if h}
        if self.title not in headers:
            raise ValueError(
                f"Required catalog title column {self.title!r} not found in CSV. "
                f"Available columns: {sorted(headers)}"
            )


def detect_schema_kind(fieldnames: list[str] | None) -> SchemaKind:
    """Auto-detect catalog vs winemag from headers."""
    if not fieldnames:
        raise ValueError("CSV has no header row.")
    headers = {h.strip() for h in fieldnames if h}
    if CATALOG_TITLE in headers or ("BRAND" in headers and "TYPES" in headers):
        return SchemaKind.CATALOG
    if "description" in headers:
        return SchemaKind.WINEMAG
    raise ValueError(
        "Unrecognized CSV schema. Expected Living Liquidz catalog "
        f"({CATALOG_TITLE!r}) or winemag ({'description'!r}). "
        f"Available columns: {sorted(headers)}"
    )


def category_from_filename(filename: str) -> str:
    """combined_domestic_wine.csv → domestic_wine."""
    stem = filename.rsplit(".", 1)[0]
    prefix = "combined_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem or filename


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Strip currency symbols / commas common in Price (₹)
    cleaned = (
        text.replace("₹", "")
        .replace(",", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
