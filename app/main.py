import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from orchestrator import answer_question
from tools.knowledge import (
    TEMPORARY_COLLECTION_DAYS,
    add_documents,
    cleanup_expired_temporary_collections,
    delete_temporary_documents,
    ingest_documents,
)
from tools.page_fetch import fetch_page
from tools.web_search import search_web


app = FastAPI()
_logger = logging.getLogger(__name__)
_cleanup_task = None


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    chat_id: str = Field(min_length=1, max_length=100)
    web_search_enabled: bool = True
    history: list[ChatTurn] = Field(default_factory=list)
    client_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    client_timezone: str | None = Field(default=None, max_length=100)


async def periodic_temporary_collection_cleanup():
    while True:
        await asyncio.to_thread(cleanup_expired_temporary_collections)
        await asyncio.sleep(60 * 60)


def report_cleanup_failure(task):
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        _logger.error(
            "Scheduled temporary collection cleanup stopped",
            exc_info=(type(error), error, error.__traceback__),
        )


@app.on_event("startup")
async def start_temporary_collection_cleanup():
    global _cleanup_task
    _cleanup_task = asyncio.create_task(periodic_temporary_collection_cleanup())
    _cleanup_task.add_done_callback(report_cleanup_failure)


@app.on_event("shutdown")
async def stop_temporary_collection_cleanup():
    if _cleanup_task is not None and not _cleanup_task.done():
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task


@app.post("/ingest")
def ingest():
    return ingest_documents()


@app.post("/ingest-url")
def ingest_url(url: str, temporary: bool = True):
    document = fetch_page(url)
    chunk_count = add_documents([document], temporary=temporary)
    result = {
        "url": document.metadata["source"],
        "chunks": chunk_count,
        "collection": "temporary" if temporary else "persistent",
    }
    if temporary:
        result["collection_date"] = datetime.now(timezone.utc).date().isoformat()
        result["retention_days"] = TEMPORARY_COLLECTION_DAYS
    return result


@app.post("/search")
def web_search(q: str):
    return {"results": search_web(q)}


@app.delete("/temporary")
def clear_temporary_documents():
    return delete_temporary_documents()


@app.get("/")
def chat_ui():
    return FileResponse(Path(__file__).with_name("ui.html"))


@app.post("/ask")
def ask(q: str, chat_id: str | None = None, web_search_enabled: bool = True):
    return answer_question(q, chat_id, web_search_enabled)


@app.post("/chat")
def chat(request: ChatRequest):
    return answer_question(
        request.message,
        chat_id=request.chat_id,
        web_search_enabled=request.web_search_enabled,
        history=request.history,
        client_date=request.client_date,
        client_timezone=request.client_timezone,
    )
