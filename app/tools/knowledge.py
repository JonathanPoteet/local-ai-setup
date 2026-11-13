import hashlib
import os
import re
import time
from datetime import datetime, timedelta, timezone

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from runtime import EMBED_URL, model_ids


TEMPORARY_COLLECTION_DAYS = 7
MIN_KNOWLEDGE_RELEVANCE = 0.6

_stores = {}
_chroma_client = chromadb.PersistentClient(path="/data")


def temporary_collection_prefix(model):
    digest = hashlib.sha256(model.encode()).hexdigest()[:32]
    return f"temp-{digest}-"


def store(temporary=False, collection_date=None):
    model = os.environ.get("EMBED_MODEL") or model_ids(EMBED_URL)[-1]
    if temporary:
        if collection_date is None:
            collection_date = datetime.now(timezone.utc).date()
        name = f"{temporary_collection_prefix(model)}{collection_date:%Y%m%d}"
    else:
        name = re.sub(r"[^a-zA-Z0-9_-]", "-", f"docs-{model}")[:63]
    key = (model, name)
    if key not in _stores:
        embeddings = OpenAIEmbeddings(
            base_url=EMBED_URL,
            api_key="x",
            model=model,
            check_embedding_ctx_length=False,
        )
        _stores[key] = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory="/data",
        )
    return _stores[key]


def _delete_cached_store(name):
    for key in [key for key in _stores if key[1] == name]:
        _stores.pop(key, None)


def cleanup_expired_temporary_collections():
    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=TEMPORARY_COLLECTION_DAYS)
    deleted_collections = 0

    for collection in _chroma_client.list_collections():
        name = collection.name
        if not name.startswith("temp-"):
            continue
        date_text = name[-8:]
        if not re.fullmatch(r"\d{8}", date_text):
            if not re.fullmatch(r"temp-[a-f0-9]{32}", name):
                continue
            legacy = _chroma_client.get_collection(name=name)
            expired = legacy.get(
                where={"expires_at": {"$lte": time.time()}},
                include=["metadatas"],
            )
            expired_ids = expired.get("ids", [])
            if expired_ids:
                legacy.delete(ids=expired_ids)
            if legacy.count() == 0:
                _chroma_client.delete_collection(name=name)
                _delete_cached_store(name)
                deleted_collections += 1
            continue
        prefix = name[:-8]
        if not re.fullmatch(r"temp-[a-f0-9]{32}-", prefix):
            continue
        try:
            collection_date = datetime.strptime(date_text, "%Y%m%d").date()
        except ValueError:
            continue
        if collection_date <= cutoff_date:
            _chroma_client.delete_collection(name=name)
            _delete_cached_store(name)
            deleted_collections += 1

    return deleted_collections


def add_documents(docs, temporary=False):
    collection_date = datetime.now(timezone.utc).date() if temporary else None
    if temporary:
        cleanup_expired_temporary_collections()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    ).split_documents(docs)
    unique = {}
    for chunk in chunks:
        chunk_id = hashlib.sha1(
            (chunk.metadata.get("source", "") + chunk.page_content).encode()
        ).hexdigest()
        unique[chunk_id] = chunk
    if unique:
        store(temporary=temporary, collection_date=collection_date).add_documents(
            list(unique.values()),
            ids=list(unique.keys()),
        )
    return len(unique)


def ingest_documents():
    documents = []
    for pattern in ("**/*.md", "**/*.txt"):
        documents += DirectoryLoader(
            "/docs",
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ).load()
    if not documents:
        return {"error": "no .md or .txt files found in /docs", "files": 0, "chunks": 0}
    chunk_count = add_documents(documents)
    return {
        "files": len(documents),
        "chunks": chunk_count,
        "collection": "persistent",
    }


def search_temporary_documents(query, limit=4):
    model = os.environ.get("EMBED_MODEL") or model_ids(EMBED_URL)[-1]
    prefix = temporary_collection_prefix(model)
    available = {collection.name for collection in _chroma_client.list_collections()}
    today = datetime.now(timezone.utc).date()
    collection_dates = [
        today - timedelta(days=day_offset)
        for day_offset in range(TEMPORARY_COLLECTION_DAYS)
        if f"{prefix}{today - timedelta(days=day_offset):%Y%m%d}" in available
    ]
    if not collection_dates:
        return []
    embedding_store = store(temporary=True, collection_date=collection_dates[0])
    query_embedding = embedding_store.embeddings.embed_query(query)
    scored_documents = []
    for collection_date in collection_dates:
        daily_store = store(temporary=True, collection_date=collection_date)
        scored_documents.extend(
            daily_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=limit,
            )
        )
    scored_documents = [
        result
        for result in scored_documents
        if result[1] >= MIN_KNOWLEDGE_RELEVANCE
    ]
    scored_documents.sort(key=lambda result: result[1], reverse=True)
    return [document for document, _score in scored_documents[:limit]]


def delete_temporary_documents():
    names = [
        collection.name
        for collection in _chroma_client.list_collections()
        if collection.name.startswith("temp-")
    ]
    for name in names:
        _chroma_client.delete_collection(name=name)
        _delete_cached_store(name)
    return {"deleted": True, "collections_deleted": len(names)}


def persistent_documents(query, limit=4):
    results = store().similarity_search_with_relevance_scores(query, k=limit)
    return [
        document
        for document, score in results
        if score >= MIN_KNOWLEDGE_RELEVANCE
    ]


def all_document_sources(persistent_hits, temporary_hits):
    return [
        hit.metadata.get("source")
        for hit in persistent_hits + temporary_hits
    ]
