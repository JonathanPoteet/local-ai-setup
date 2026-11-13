import base64
import re
import requests
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit
from xml.etree import ElementTree

from fastapi import HTTPException


LETTER_REQUEST = re.compile(
    r"^\s*(?:write|draft|compose)\s+(?:me\s+)?(?:a|an)\s+letter\s+about\b",
    re.I,
)


class SearchResultsParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.capture = None
        self.capture_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            result = {
                "title": "",
                "url": urljoin("https://duckduckgo.com", attributes.get("href", "")),
                "snippet": "",
            }
            self.results.append(result)
            self.capture = "title"
            self.capture_depth = 1
        elif "result__snippet" in classes and self.results:
            self.capture = "snippet"
            self.capture_depth = 1
        elif self.capture:
            self.capture_depth += 1

    def handle_endtag(self, _):
        if self.capture:
            self.capture_depth -= 1
            if self.capture_depth <= 0:
                self.capture = None

    def handle_data(self, data):
        if self.capture and self.results:
            self.results[-1][self.capture] += data


class BingResultsParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.current = None
        self.capture = None
        self.capture_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class", "").split()
        if tag == "li" and "b_algo" in classes:
            self.current = {"title": "", "url": "", "snippet": ""}
            self.results.append(self.current)
            self.capture = None
        elif self.current:
            if tag == "h2":
                self.capture = "title"
                self.capture_depth = 1
            elif tag == "p" and not self.current["snippet"]:
                self.capture = "snippet"
                self.capture_depth = 1
            elif self.capture:
                self.capture_depth += 1
            if tag == "a" and self.capture == "title":
                self.current["url"] = attributes.get("href", "")

    def handle_endtag(self, tag):
        if tag == "li":
            self.current = None
            self.capture = None
        elif self.capture:
            self.capture_depth -= 1
            if self.capture_depth <= 0:
                self.capture = None

    def handle_data(self, data):
        if self.capture and self.current:
            self.current[self.capture] += data


def parse_duckduckgo_results(html, limit):
    parser = SearchResultsParser()
    parser.feed(html)
    results = []
    seen_urls = set()
    for item in parser.results:
        result = {key: value.strip() for key, value in item.items()}
        redirect_url = urlsplit(result["url"])
        if redirect_url.hostname and redirect_url.hostname.endswith("duckduckgo.com"):
            result["url"] = parse_qs(redirect_url.query).get("uddg", [result["url"]])[0]
        if result["url"] and result["title"] and result["url"] not in seen_urls:
            results.append(result)
            seen_urls.add(result["url"])
        if len(results) >= limit:
            break
    return results


def parse_bing_results(html, limit):
    parser = BingResultsParser()
    parser.feed(html)
    results = []
    seen_urls = set()
    for result in parser.results:
        result = {key: value.strip() for key, value in result.items()}
        parsed_url = urlsplit(result["url"])
        if parsed_url.hostname and parsed_url.hostname.endswith("bing.com"):
            encoded_url = parse_qs(parsed_url.query).get("u", [""])[0]
            if encoded_url.startswith("a1"):
                try:
                    decoded_url = base64.urlsafe_b64decode(
                        encoded_url[2:] + "=" * (-len(encoded_url[2:]) % 4)
                    ).decode("utf-8")
                    if decoded_url.startswith(("http://", "https://")):
                        result["url"] = decoded_url
                except (ValueError, UnicodeDecodeError):
                    pass
        if result["url"] and result["title"] and result["url"] not in seen_urls:
            results.append(result)
            seen_urls.add(result["url"])
        if len(results) >= limit:
            break
    return results


def is_broad_news_request(question):
    return bool(re.search(
        r"\b("
        r"(?:latest|top|breaking|current|today'?s?)\s+(?:world\s+|global\s+)?news|"
        r"news\s+(?:today|headlines)|(?:latest\s+)?headlines"
        r")\b",
        question,
        re.IGNORECASE,
    ))


def normalize_search_query(query):
    query = re.sub(
        r"\b(?:"
        r"please|can\s+you|could\s+you|would\s+you|"
        r"tell\s+me(?:\s+about)?|search(?:\s+the)?\s+web(?:\s+for)?|"
        r"search\s+for|look\s+up|find\s+out(?:\s+about)?"
        r")\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"\b(?:a|an|the|is|are|what|who|where|when|does|do|about|for|"
        r"and|to|of|me|you|it|this|that)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    return " ".join(query.split()).strip(" .?!,;:")


def is_letter_request(query):
    return bool(LETTER_REQUEST.match(query))


def run_search(query, current_date, limit=5):
    if is_letter_request(query):
        match = re.search(r"\babout\s+(.+)$", query, re.I)
        if match:
            query = match.group(1).strip(" .?!,;:")
    try:
        return search_web(normalize_search_query(query), limit, current_date), None
    except HTTPException as exc:
        return [], str(exc.detail)


def format_excerpts(results, limit=5):
    return "\n\n".join(
        f"- **{result['title']}**\n  {result['snippet']}\n  {result['url']}"
        for result in results[:limit]
    )


def search_google_news(query, limit=5, current_date=None):
    current_date = current_date or datetime.now().astimezone().date()
    broad_news = is_broad_news_request(query)
    endpoint = (
        "https://news.google.com/rss"
        if broad_news else "https://news.google.com/rss/search"
    )
    params = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    if not broad_news:
        params["q"] = f"{query} after:{current_date - timedelta(days=1)}"
    response = requests.get(
        endpoint,
        params=params,
        headers={"User-Agent": "Mozilla/5.0 (compatible; LocalSetup/1.0)"},
        timeout=(5, 15),
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    results = []
    seen_urls = set()
    for item in root.findall(".//item"):
        title = item.findtext("title", default="").strip()
        url = item.findtext("link", default="").strip()
        snippet = re.sub(r"<[^>]+>", " ", item.findtext("description", default=""))
        snippet = " ".join(unescape(snippet).split())
        published = item.findtext("pubDate", default="").strip()
        try:
            published_date = parsedate_to_datetime(published).date()
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            published_date < current_date - timedelta(days=1)
            or published_date > current_date + timedelta(days=1)
        ):
            continue
        if title and url and url not in seen_urls:
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "published": published,
            })
            seen_urls.add(url)
        if len(results) >= limit:
            break
    return results


def search_web(query, limit=5, current_date=None):
    query = normalize_search_query(query) or query
    if is_broad_news_request(query):
        try:
            results = search_google_news(query, limit, current_date)
            if results:
                return results
        except (requests.RequestException, ElementTree.ParseError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Current news search failed: {exc}",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="Current news search returned no recent articles",
        )

    errors = []
    providers = (
        ("DuckDuckGo", "https://html.duckduckgo.com/html/", parse_duckduckgo_results),
        ("Bing", "https://www.bing.com/search", parse_bing_results),
    )
    for provider, endpoint, parser_function in providers:
        try:
            response = requests.get(
                endpoint,
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; LocalSetup/1.0)"},
                timeout=(5, 15),
            )
            response.raise_for_status()
            if response.status_code != 200:
                errors.append(f"{provider} returned HTTP {response.status_code}")
                continue
            results = parser_function(response.text, limit)
            if results:
                return results
            errors.append(f"{provider} returned no parseable results")
        except requests.RequestException as exc:
            errors.append(f"{provider}: {exc}")

    try:
        results = search_google_news(query, limit, current_date)
        if results:
            return results
        errors.append("Google News returned no parseable results")
    except (requests.RequestException, ElementTree.ParseError) as exc:
        errors.append(f"Google News: {exc}")

    raise HTTPException(status_code=502, detail="All web search providers failed: " + "; ".join(errors))
