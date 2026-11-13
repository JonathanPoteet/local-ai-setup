import os
import requests
from langchain_openai import ChatOpenAI


CHAT_URL = "http://127.0.0.1:8081/v1"
EMBED_URL = "http://127.0.0.1:8082/v1"

_llm = None


def model_ids(url):
    response = requests.get(f"{url}/models", timeout=10)
    response.raise_for_status()
    return [model["id"] for model in response.json()["data"]]


def chat_model():
    global _llm
    if _llm is None:
        model = os.environ.get("CHAT_MODEL") or model_ids(CHAT_URL)[0]
        _llm = ChatOpenAI(
            base_url=CHAT_URL,
            api_key="x",
            model=model,
            temperature=0.2,
            max_tokens=512,
            timeout=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _llm
