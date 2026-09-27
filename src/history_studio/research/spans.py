"""Deterministic evidence spans over a bounded, normalized read representation."""
import hashlib
import json
import re
from dataclasses import dataclass

from history_studio.models.research import EvidenceReference
from .web_tools import normalize_text


@dataclass(frozen=True)
class SourceSpans:
    source_id: str
    source_version: str
    text: str
    truncated: bool
    spans: dict[str, tuple[int, int]]

    def observation(self) -> dict:
        return {"source_id": self.source_id, "source_version": self.source_version,
                "truncated": self.truncated,
                "spans": [{"span_id": key, "text": self.text[start:end]}
                          for key, (start, end) in self.spans.items()]}


def make_spans(source_id: str, text: str, truncated: bool = False) -> SourceSpans:
    text = normalize_text(text)
    # Algorithm version and truncation are part of representation identity.
    digest = hashlib.sha256(json.dumps(["spans-v1", source_id, text, truncated],
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    version = "VER-" + digest
    spans = {}
    start = 0
    while start < len(text):
        ceiling = min(start + 600, len(text))
        segment = text[start:ceiling]
        # Pack sentences within 600 code points; preserve punctuation verbatim.
        boundaries = [m.end() for m in re.finditer(r'[。！？!?][”’"\']*|\.(?=\s|$)', segment)]
        end = start + boundaries[-1] if boundaries else ceiling
        if not boundaries and ceiling < len(text):
            space = text.rfind(" ", start + 300, ceiling)
            if space != -1:
                end = space
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            key = "SPAN-" + hashlib.sha256(f"{version}:{start}:{end}".encode()).hexdigest()
            spans[key] = (start, end)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return SourceSpans(source_id, version, text, truncated, spans)


def resolve_selection(source_id: str, span_id: str, reads: dict[str, SourceSpans],
                      seen: dict[str, tuple[str, str]], location: list) -> EvidenceReference:
    from .diagnostics import ArtifactValidationError
    code = None
    read = reads.get(source_id)
    owner = seen.get(span_id)
    if read is None:
        code = "source_not_read"
    elif owner is not None and owner[0] != source_id:
        code = "span_source_mismatch"
    elif owner is not None and owner[1] != read.source_version:
        code = "stale_source_span"
    elif span_id not in read.spans:
        code = "span_not_found"
    if code:
        raise ArtifactValidationError(code, location, selection={
            "source_id": source_id, "span_id": span_id,
            "source_was_read": read is not None,
            "current_source_version": read.source_version if read else None,
            "selected_source_version": owner[1] if owner else None})
    start, end = read.spans[span_id]
    return EvidenceReference(source_id=source_id, span_id=span_id,
        source_version=read.source_version, excerpt=read.text[start:end])
