"""Bounded evidence mismatch observations; never used to accept an excerpt."""
import hashlib
import json
import unicodedata
from difflib import SequenceMatcher

from history_studio.models.base import Contract
from .web_tools import normalize_text


class EvidenceMismatch(Contract):
    source_id: str
    proposed_preview: str | None
    normalized_preview: str | None
    proposed_length: int
    normalized_length: int
    proposed_sha256: str
    normalized_sha256: str
    source_was_read: bool
    retained_text_length: int
    raw_text_length: int
    source_truncated: bool | None
    occurs_before_normalization: bool
    occurs_after_normalization: bool
    nearest_span_preview: str | None
    nearest_span_start: int | None
    nearest_span_end: int | None
    category: str
    boundary_prefix_match: bool


def evidence_mismatch(source_id: str, excerpt: str, passages: dict[str, str], *,
                      raw_text: str | None = None, truncated: bool | None = None) -> EvidenceMismatch:
    from .diagnostics import safe_provider_message, safe_provider_field

    def preview(value: str) -> str | None:
        # JSON escapes expose whitespace differences without terminal control characters.
        return safe_provider_message(json.dumps(value[:240], ensure_ascii=False))

    page = passages.get(source_id, "")
    raw = page if raw_text is None else raw_text
    quote = normalize_text(excerpt)
    before = bool(excerpt) and excerpt in raw
    after = bool(quote) and quote in page
    category = "unknown"
    boundary = False
    start = end = None
    span = ""
    if source_id not in passages:
        category = "source_id_mismatch"
    elif after:
        category = "exact" if before else "whitespace_only"
    elif quote and unicodedata.normalize("NFC", quote) in unicodedata.normalize("NFC", page):
        category = "unicode_normalization_difference"
    else:
        def without_punctuation(value):
            return "".join(c for c in value if not unicodedata.category(c).startswith("P"))
        stripped = without_punctuation(quote)
        if stripped and stripped in without_punctuation(page):
            category = "punctuation_difference"
    if page and quote:
        match = SequenceMatcher(None, quote, page, autojunk=False).find_longest_match()
        if match.size:
            start = max(0, match.b - match.a)
            # Small sources are not copied in full either. This is a candidate, not evidence.
            end = min(len(page), start + min(240, max(1, len(page) // 2), len(quote) + 32))
            span = page[start:end]
        boundary = bool(truncated and any(page.endswith(quote[:n])
                        for n in range(min(8, len(quote)), min(len(quote), len(page)) + 1)
                        if n >= 4 and n < len(quote)))
    return EvidenceMismatch(source_id=safe_provider_field(source_id) or "<redacted>", proposed_preview=preview(excerpt),
        normalized_preview=preview(quote), proposed_length=len(excerpt), normalized_length=len(quote),
        proposed_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
        normalized_sha256=hashlib.sha256(quote.encode()).hexdigest(),
        source_was_read=source_id in passages, retained_text_length=len(page), raw_text_length=len(raw),
        source_truncated=truncated, occurs_before_normalization=before, occurs_after_normalization=after,
        nearest_span_preview=preview(span) if span else None, nearest_span_start=start, nearest_span_end=end,
        category=category, boundary_prefix_match=boundary)
