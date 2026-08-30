"""Unit tests for recommendation text cleanup."""

from __future__ import annotations

import re

from rag.chain import to_plain_professional_text

SAMPLE_MARKDOWN = """
**Best match for a "strong" wine:**

| Wine | Key Features | Why it's a fit |
|------|--------------|----------------|
| **RIO RED STRONG PORT WINE** | • Fortified port <br>• ₹140 | • The word STRONG is in the title |

---

### Quick tasting note
- **Appearance**: Deep ruby.
- **Aroma**: Blackberries and chocolate.

### Bottom line
If you're after a wine that's unmistakably "strong", choose **Rio Red Strong Port Wine**. Enjoy responsibly!
"""

TRIO_MARKDOWN = """
**Your "strong" wine trio**

| # | Wine | Origin | Grapes | Body | Price (€) | Why it's "strong" |
|---|------|--------|--------|------|-----------|-------------------|
| 1 | **SCAIA Torre Mellotti Cabernet Sauvignon** | Italy | Cabernet Sauvignon | Full-bodied | 4195 | Classic high-tannin Cab |

> **Why these three?**
> • SCAIA is bold.

### Pairing & Service Tips

| Wine | Pairing |
|------|---------|
| **SCAIA** | Grilled lamb |

### Quick Summary
- SCAIA – go-to strong Cab.

Enjoy your tasting adventure!
"""


def test_strips_tables_and_section_headers() -> None:
    plain = to_plain_professional_text(SAMPLE_MARKDOWN)
    assert "|" not in plain
    assert "###" not in plain
    assert "**" not in plain
    assert "Quick tasting note" not in plain
    assert "Bottom line" not in plain
    assert re.search(r"(?im)^appearance\s*:", plain) is None
    assert re.search(r"(?im)^aroma\s*:", plain) is None
    assert "Enjoy responsibly" not in plain
    assert "Best match" not in plain
    assert "RIO RED STRONG PORT WINE" in plain or "Rio Red Strong Port Wine" in plain
    assert "<br" not in plain.lower()


def test_strips_trio_blog_format() -> None:
    plain = to_plain_professional_text(TRIO_MARKDOWN)
    assert "|" not in plain
    assert "**" not in plain
    assert "wine trio" not in plain.lower()
    assert "Why these three" not in plain
    assert "Pairing &" not in plain
    assert "Service Tips" not in plain
    assert "Quick Summary" not in plain
    assert "tasting adventure" not in plain.lower()
    assert "SCAIA" in plain


def test_live_style_recommendation_block() -> None:
    raw = """
**Recommendation:**
**RIO RED STRONG PORT WINE**

- **Type:** Fortified wine (port)
- **Origin:** India
- **Volume:** 330 ml
- **Price:** Rs 140.00

This is the only wine that carries a strong designation.
"""
    plain = to_plain_professional_text(raw)
    assert "*" not in plain
    assert "|" not in plain
    assert "Recommendation" not in plain
    assert "Type:" not in plain
    assert "RIO RED STRONG PORT WINE" in plain
    assert "India" in plain or "140" in plain
