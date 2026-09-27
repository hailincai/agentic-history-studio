import socket

import pytest

from history_studio.research.web_tools import (
    PageText, SourceReadError, _fetch, normalize_text, read_public_source, source_reference,
)


def test_html_extraction_preserves_source_words_and_drops_scripts(monkeypatch) -> None:
    html = "<html><title>历史</title><script>invented()</script><p>李白 生于 701 年。</p></html>"
    monkeypatch.setattr("history_studio.research.web_tools._fetch", lambda url: (url, html))
    result = read_public_source(source_reference("https://example.org/"), 1000)
    assert "李白 生于 701 年。" in result.text
    assert "invented" not in result.text
    assert result.sources[0].title == "历史"
    assert result.source_id == result.sources[0].source_id


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"])
def test_source_reader_blocks_private_addresses(monkeypatch, address) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, 1, 6, "", (address, 443))])
    with pytest.raises(SourceReadError, match="non_public"):
        _fetch("https://example.org/")


def test_source_content_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr("history_studio.research.web_tools._fetch", lambda url: (url, "<p>" + "x" * 5000 + "</p>"))
    result = read_public_source(source_reference("https://example.org/"), 1000)
    assert result.truncated
    assert len(result.text) == 1000
