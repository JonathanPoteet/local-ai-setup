import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests
from fastapi import HTTPException
from langchain_core.documents import Document


MAX_URL_BYTES = 2 * 1024 * 1024
MAX_SEARCH_PAGE_INSPECTIONS = 5
MAX_SEARCH_PAGE_EXCERPT_CHARS = 1500


class PageTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored_tags = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.ignored_tags.append(tag)
        elif not self.ignored_tags:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.ignored_tags:
            if tag == self.ignored_tags[-1]:
                self.ignored_tags.pop()
        else:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.ignored_tags:
            self.parts.append(data)


def validate_public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail="URL must use http or https and include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(status_code=400, detail="URLs with embedded credentials are not allowed")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid URL port") from exc
    if port not in (80, 443):
        raise HTTPException(status_code=400, detail="Only ports 80 and 443 are allowed")

    hostname = parsed.hostname.rstrip(".")
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            ]
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="URL hostname could not be resolved") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise HTTPException(status_code=400, detail="URL must resolve only to public IP addresses")
    return parsed


def fetch_page(url):
    current_url = url
    for redirect_count in range(6):
        validate_public_url(current_url)
        try:
            with requests.get(
                current_url,
                headers={"User-Agent": "LocalSetup/1.0"},
                timeout=(5, 15),
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.is_redirect:
                    if redirect_count == 5:
                        raise HTTPException(status_code=400, detail="Too many URL redirects")
                    location = response.headers.get("Location")
                    if not location:
                        raise HTTPException(status_code=502, detail="Redirect response has no destination")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type not in ("text/html", "text/plain"):
                    raise HTTPException(status_code=415, detail="URL must return HTML or plain text")
                body = bytearray()
                for part in response.iter_content(chunk_size=65536):
                    body.extend(part)
                    if len(body) > MAX_URL_BYTES:
                        raise HTTPException(status_code=413, detail="Page exceeds the 2 MiB size limit")
                encoding = response.encoding or "utf-8"
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Could not fetch URL: {exc}") from exc

        text = bytes(body).decode(encoding, errors="replace")
        if content_type == "text/html":
            parser = PageTextParser()
            parser.feed(text)
            text = " ".join(" ".join(parser.parts).split())
        else:
            text = " ".join(text.split())
        if not text:
            raise HTTPException(status_code=422, detail="URL contained no extractable text")
        return Document(page_content=text, metadata={"source": current_url, "type": "url"})
    raise HTTPException(status_code=400, detail="Too many URL redirects")


def inspect_search_results(results, limit=MAX_SEARCH_PAGE_INSPECTIONS):
    inspected_results = []
    for result in results:
        inspected = dict(result)
        inspected["page_inspected"] = False
        inspected["page_inspection_error"] = None
        if len(inspected_results) >= limit or not result.get("url"):
            inspected_results.append(inspected)
            continue
        try:
            page = fetch_page(result["url"])
        except HTTPException as exc:
            inspected["page_inspection_error"] = str(exc.detail)
        else:
            excerpt = page.page_content[:MAX_SEARCH_PAGE_EXCERPT_CHARS]
            inspected["snippet"] = "Page content: " + excerpt
            if len(page.page_content) > MAX_SEARCH_PAGE_EXCERPT_CHARS:
                inspected["snippet"] += "…"
            inspected["url"] = page.metadata["source"]
            inspected["page_inspected"] = True
        inspected_results.append(inspected)
    return inspected_results
