"""Tool for returning source excerpts for direct reference lookups."""
import re
from datetime import date

from fastapi import HTTPException

from tools.web_search import format_excerpts, normalize_search_query, search_web


_REFERENCE_LOOKUP = re.compile(r"^\s*tell\s+me\s+about\b", re.I)


def matches(question):
    return bool(_REFERENCE_LOOKUP.match(question))


def run(question: str, today: date):
    try:
        results = search_web(normalize_search_query(question), 5, today)
        error = None
    except HTTPException as exc:
        results = []
        error = str(exc.detail)

    answer = (
        "Here are relevant excerpts from the search results:\n\n" + format_excerpts(results, 3)
        if results else
        "I couldn't retrieve reliable source excerpts for this question"
        + (f": {error}" if error else ".")
    )
    return answer, results[:3], error
