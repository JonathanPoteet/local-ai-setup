Project overview
This is an exploratory, local-first AI setup built with Podman-compatible containers. It combines local chat and embedding models with a small web app, to explore what a self-hosted assistant could look like—not a production-ready deployment.

The setup currently includes:

llama.cpp servers for chat and embeddings, using GPU access.
FastAPI and a browser UI for chat, URL ingestion, document indexing, and web search.
ChromaDB for persistent knowledge and temporary documents that expire after seven days.
Tailscale for network access between services.
Search tools that can retrieve current news and web results when enabled.
Model files and documents are mounted from the host. The Tailscale auth key is supplied via .env; use .env.example as a template and keep .env private.

The current Compose configuration assumes a Windows/WSL-style environment with model files in a external HD, D:/AI-Models and GPU access through /dev/dxg. These paths and device settings will likely need adjustment on other systems.
