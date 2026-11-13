"""Tool for returning current headline results without model generation."""
from datetime import date

from fastapi import HTTPException

from tools.web_search import is_broad_news_request, search_web


def matches(question):
    return is_broad_news_request(question)


def run(today: date, enabled: bool):
    results = []
    error = None
    if enabled:
        try:
            results = search_web("latest news", 5, today)
        except HTTPException as exc:
            error = str(exc.detail)

    if results:
        lines = [
            f"- {result['title']}"
            + (f" ({result['published']})" if result.get("published") else "")
            + (f"\n  {result['snippet']}" if result.get("snippet") else "")
            for result in results
        ]
        answer = f"Recent headlines found as of {today.isoformat()}:\n\n" + "\n\n".join(lines)
    elif not enabled:
        answer = "Turn on web search to retrieve current headlines."
    else:
        answer = (
            f"I couldn't retrieve verified recent headlines for {today.isoformat()}"
            + (f": {error}" if error else ".")
        )

    return answer, results, error
