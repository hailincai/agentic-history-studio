import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from history_studio.models.sources import SourceReference, SourceType
from .boundaries import ToolObservation


class SourceReadError(RuntimeError):
    pass


def canonical_url(url: str) -> str:
    from pydantic import HttpUrl, TypeAdapter
    parsed = urlsplit(str(TypeAdapter(HttpUrl).validate_python(url)))
    if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
        raise ValueError("URL credentials and nonstandard ports are not allowed")
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")]
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path or "/", urlencode(sorted(query)), ""))


def source_reference(url: str, title: str | None = None) -> SourceReference:
    url = canonical_url(url)
    domain = urlsplit(url).hostname
    return SourceReference(source_id="SRC-" + hashlib.sha256(url.encode()).hexdigest()[:20],
                           url=url, title=title or domain or url, domain=domain,
                           source_type=SourceType.UNKNOWN, accessed_at=datetime.now(timezone.utc))


class PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "template"):
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "td"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "template"):
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False
        if tag in ("p", "div", "li", "h1", "h2", "h3", "td"):
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)
            if self.in_title:
                self.title.append(data)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def _fetch(url: str, max_bytes: int = 1_000_000) -> tuple[str, str]:
    """Public-address-only, IP-pinned fetch. Recheck every redirect; no proxies/cookies."""
    for _ in range(5):
        parsed = urlsplit(canonical_url(url))
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise SourceReadError("non_public_source_address")
        # Connect only to the resolved, validated address; retain TLS hostname verification.
        address = addresses[0][4][0]
        connection = http.client.HTTPConnection(host, port, timeout=20)
        try:
            sock = socket.create_connection((address, port), timeout=20)
            if parsed.scheme == "https":
                try:
                    sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
                except Exception:
                    sock.close()
                    raise
            connection.sock = sock
            target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            connection.request("GET", target, headers={"User-Agent": "AgenticHistoryStudio/0.2 research",
                                                       "Accept-Encoding": "identity"})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise SourceReadError("redirect_without_location")
                url = canonical_url(urljoin(url, location))
                continue
            if response.status != 200:
                raise SourceReadError(f"source_http_{response.status}")
            content_type = response.getheader("Content-Type", "")
            if not any(t in content_type.lower() for t in ("text/html", "text/plain", "application/xhtml+xml")):
                raise SourceReadError("unsupported_source_content_type")
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise SourceReadError("source_size_limit")
            charset = re.search(r"charset=[\"']?([\w-]+)", content_type, flags=re.I)
            encoding = charset.group(1) if charset else "utf-8"
            # HTML's declared charset is useful for historical Chinese sites.
            if not charset:
                meta = re.search(br"charset=[\"']?([\w-]+)", raw[:4096], flags=re.I)
                if meta:
                    encoding = meta.group(1).decode("ascii")
            return canonical_url(url), raw.decode(encoding, errors="replace")
        finally:
            connection.close()
    raise SourceReadError("redirect_limit")


def read_public_source(source: SourceReference, max_chars: int) -> ToolObservation:
    final_url, html = _fetch(str(source.url))
    parser = PageText()
    parser.feed(html)
    text = normalize_text("".join(parser.parts))
    if not text:
        raise SourceReadError("empty_source")
    # Keep the discovered identity stable across redirects; record final URL as metadata.
    updated = source.model_copy(update={"title": normalize_text("".join(parser.title)) or source.title,
                                        "accessed_at": datetime.now(timezone.utc),
                                        "notes": f"Retrieved from {final_url}"})
    return ToolObservation(kind="source", sources=[updated], source_id=source.source_id,
                           text=text[:max_chars], truncated=len(text) > max_chars)
