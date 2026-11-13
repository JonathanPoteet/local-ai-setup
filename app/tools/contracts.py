import json


def serialize_tool_result(results=(), error=None):
    normalized_results = [
        {
            "title": result.get("title") or result.get("source") or "Local document",
            "url": result.get("url"),
            "snippet": result.get("snippet") or result.get("content", ""),
            "source": result.get("source"),
            "published": result.get("published"),
            "page_inspected": result.get("page_inspected", False),
            "page_inspection_error": result.get("page_inspection_error"),
        }
        for result in results
    ]
    return json.dumps(
        {
            "status": "error" if error else "ok",
            "results": normalized_results,
            "error": error,
        },
        ensure_ascii=False,
    )
