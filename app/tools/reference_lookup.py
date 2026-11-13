"""Look up concise reference summaries from Wikipedia's public REST API."""
from html.parser import HTMLParser
from urllib.parse import quote

import requests
from fastapi import HTTPException


_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_USER_AGENT = "LocalSetup/1.0 (local reference lookup)"


class _ExcerptParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain_excerpt(value):
    parser = _ExcerptParser()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


def search_references(query: str):
    try:
        response = requests.get(
            _SEARCH_URL,
            params={"q": query.strip(), "limit": 5},
            headers={"User-Agent": _USER_AGENT},
            timeout=(5, 15),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Reference lookup failed: {exc}",
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise HTTPException(
            status_code=502,
            detail="Reference lookup returned an invalid response",
        )

    results = []
    for page in payload["pages"][:5]:
        if not isinstance(page, dict):
            continue
        title = page.get("title", "").strip()
        key = page.get("key", "").strip()
        if not title or not key:
            continue
        excerpt = _plain_excerpt(page.get("excerpt", ""))
        description = page.get("description", "").strip()
        snippet = " — ".join(part for part in (description, excerpt) if part)
        results.append(
            {
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{quote(key.replace(' ', '_'), safe='()_,.-')}",
                "snippet": snippet,
                "source": "Wikipedia",
                "published": None,
            }
        )
    return results
