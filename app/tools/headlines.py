"""Search recent news results for a requested topic."""
from datetime import date

from tools.web_search import search_web


def search_headlines(query: str, today: date):
    query = query.strip() or "latest news"
    if "news" not in query.lower() and "headline" not in query.lower():
        query = f"{query} news"
    return search_web(query, 5, today)
