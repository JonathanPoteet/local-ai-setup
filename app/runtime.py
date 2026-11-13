import os
import requests
from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


CHAT_URL = os.environ.get("CHAT_BASE_URL", "http://127.0.0.1:8081/v1")
EMBED_URL = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8082/v1")

_llm = None
_chat_model_id = None
_embedder = None
_embedding_model = None


def model_ids(url):
    response = requests.get(f"{url}/models", timeout=10)
    response.raise_for_status()
    return [model["id"] for model in response.json()["data"]]


def chat_model():
    global _llm
    if _llm is None:
        model = chat_model_id()
        _llm = ChatOpenAI(
            base_url=CHAT_URL,
            api_key="not-needed",
            model=model,
            temperature=0.2,
            max_tokens=512,
            timeout=600,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _llm


def chat_model_id():
    global _chat_model_id
    configured_model = os.environ.get("CHAT_MODEL") or os.environ.get("SMALL_MODEL")
    if configured_model:
        return configured_model
    if _chat_model_id is None:
        try:
            models = [model for model in model_ids(CHAT_URL) if "embed" not in model.lower()]
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Chat server unavailable: {exc}"
            ) from exc
        if not models:
            raise HTTPException(status_code=503, detail="Chat server lists no models")
        _chat_model_id = models[0]
    return _chat_model_id


def embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = os.environ.get("EMBED_MODEL") or model_ids(EMBED_URL)[-1]
    return _embedding_model


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(
            base_url=EMBED_URL,
            api_key="x",
            model=embedding_model(),
            check_embedding_ctx_length=False,
        )
    return _embedder
