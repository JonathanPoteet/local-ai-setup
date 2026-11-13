from datetime import date
import re

from fastapi import HTTPException
from langchain_core.tools import StructuredTool

from tools.contracts import serialize_tool_result
from tools.headlines import search_headlines
from tools.knowledge import search_knowledge
from tools.reference_lookup import search_references
from tools.web_search import search_web


WEB_TOOL_NAMES = frozenset({"web_search", "reference_lookup", "headlines"})
_HEADLINES_REQUEST = re.compile(
    r"\b(?:latest|recent|breaking|today'?s?|current)\b.{0,50}\b(?:news|headlines)\b|"
    r"\b(?:news|headlines)\b.{0,30}\b(?:today|latest|recent|breaking)\b",
    re.I,
)
_EXPLICIT_SEARCH = re.compile(r"\b(?:search|look\s+up|browse)\b", re.I)
_REFERENCE_REQUEST = re.compile(
    r"^\s*(?:tell\s+me\s+about|give\s+me\s+(?:an?\s+)?overview\s+of)\b",
    re.I,
)


def _create_search_tool(name, description, search):
    def run(query: str) -> str:
        try:
            results = search(query)
        except HTTPException as exc:
            return serialize_tool_result(error=str(exc.detail))
        return serialize_tool_result(results)

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=description,
    )


def create_tools(today: date, web_search_enabled=True, query=""):
    tools = [search_knowledge]
    if web_search_enabled:
        web_tools = {
            "headlines": _create_search_tool(
                "headlines",
                "Retrieve recent news headlines for a topic. Query should be a news topic.",
                lambda query: search_headlines(query, today),
            ),
            "reference_lookup": _create_search_tool(
                "reference_lookup",
                "Find source-backed background information or an overview about a subject. "
                "Query should be the subject only.",
                search_references,
            ),
            "web_search": _create_search_tool(
                "web_search",
                "Search the public web for general information or an explicitly requested "
                "online lookup. Query should be a concise search phrase.",
                lambda query: search_web(query, 5, today),
            ),
        }
        preferred_tool = preferred_web_tool(query)

        if preferred_tool:
            tools.append(web_tools[preferred_tool])
        else:
            tools.extend(web_tools.values())
    return tools


def preferred_web_tool(query):
    if _HEADLINES_REQUEST.search(query):
        return "headlines"
    if _EXPLICIT_SEARCH.search(query):
        return "web_search"
    if _REFERENCE_REQUEST.search(query):
        return "reference_lookup"
    return None
