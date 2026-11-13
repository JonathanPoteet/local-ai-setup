# Local AI Setup

> **A small, local-first AI playground built with Podman.**
>
> This project is exploratory: a hands-on sketch of what a self-hosted assistant
> might look like, not a polished product or production-ready deployment.

Run local chat and embedding models, ask questions about your own documents, and
optionally fetch current information from the web—all through a lightweight
browser interface.

## What’s inside

| Component | Role |
| --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Serves local chat and embedding models through OpenAI-compatible endpoints. |
| FastAPI | Connects the UI, model servers, knowledge store, and tools. |
| ChromaDB | Stores document embeddings for local retrieval. |
| Tailscale | Provides private access to the app and connects the services. |
| Browser UI | Offers chat, conversation history, document ingestion, and search controls. |

The chat model handles general questions. The app can also route requests to
purpose-built tools, such as web search and headline lookup. Markdown and text
files, plus text extracted from web pages, can be indexed as knowledge.

For an overview of available tools and possible follow-up work, see
[app/tools/README.md](app/tools/README.md).

## How it fits together

```text
Browser ── Tailscale ── FastAPI app
                            ├── llama.cpp chat server
                            ├── llama.cpp embedding server
                            ├── ChromaDB knowledge store
                            └── Optional public web search
```

Chat and embedding run as separate llama.cpp services. ChromaDB data is
persisted in `./chroma`; temporary URL-ingestion collections expire after
seven days. Web search uses external search providers, so that feature requires
internet access and is not fully local.

## Prerequisites

- Podman with a working Compose provider (`podman compose`).
- A compatible Windows/WSL setup with GPU passthrough configured for
  `/dev/dxg`—or the appropriate GPU configuration for your host.
- Chat model files in `D:/AI-Models`.
- Embedding model files in `D:/AI-Models/embeddings`.
- A Tailscale auth key and a Tailscale Serve configuration.

The paths and GPU device in the included Compose file are specific to the
current setup. Adjust them for your hardware and operating system.

## Getting started

1. Copy `.env.example` to `.env` and add your Tailscale auth key:

   ```dotenv
   TS_AUTHKEY=your-tailscale-auth-key
   ```

   Keep `.env` private; it is excluded by `.gitignore`.

2. Review `compose.yaml` and update the model mount paths and GPU settings as
   needed.

3. Review `serve.json` and configure the Tailscale hostname and proxy for your
   tailnet. The current example contains a project-specific hostname.

4. Start the services:

   ```powershell
   podman compose up -d
   ```

5. Open the app using the Tailscale Serve URL configured in `serve.json`.

The Python container installs its dependencies when it starts. The model
servers expose OpenAI-compatible APIs internally at ports `8081` (chat) and
`8082` (embeddings); the web app listens on port `8000`.

## Using the app

- **Chat:** Ask the local model a question; allow web search when you want
  current or external information.
- **Index local documents:** Put `.md` or `.txt` files in `docs/`, then choose
  **Index /docs**.
- **Add a web page:** Submit a URL to extract and index its text. Pages are
  temporary by default; the UI also offers persistent storage.
- **Manage temporary knowledge:** Use **Clear temporary** to remove temporary
  collections. Otherwise they expire after seven days.

The browser stores chat transcripts locally. ChromaDB data is stored in
`chroma/`, which is ignored by Git.

## Configuration

The app supports these optional environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CHAT_MODEL` | Select the chat model ID. | First model returned by the chat server. |
| `EMBED_MODEL` | Select the embedding model ID. | Last model returned by the embedding server. |
| `CHAT_BASE_URL` | Override the chat API URL. | `http://127.0.0.1:8081/v1` |
| `EMBED_BASE_URL` | Override the embedding API URL. | `http://127.0.0.1:8082/v1` |
| `SEMANTIC_ROUTING` | Set to `0` to disable semantic web-search routing. | Enabled |

These variables need to be supplied to the app container if you want to override
its defaults.

## Project layout

```text
.
├── app/
│   ├── main.py              # FastAPI routes
│   ├── orchestrator.py      # Chooses tools or the normal chat path
│   ├── runtime.py           # Shared chat and embedding clients
│   ├── tools/               # Knowledge, web search, and page-fetch tools
│   └── ui.html              # Browser chat interface
├── chroma/                  # Persistent local vector database (Git-ignored)
├── docs/                    # Markdown and text files to index
├── compose.yaml
├── serve.json               # Tailscale Serve configuration
└── .env.example             # Environment-variable template
```

## Current limitations

- This is a personal experiment; security, reliability, and deployment
  ergonomics need more work before exposing it beyond a trusted environment.
- Compose currently uses host networking, privileged Tailscale settings, and
  Windows/WSL-oriented GPU configuration.
- Model compatibility, memory requirements, and performance depend on the
  selected models and hardware.
- Web search depends on external providers and may fail or change over time.
- The Tailscale auth key and Serve hostname must be configured for your own
  tailnet.

Contributions, experiments, and ideas for a better local AI setup are welcome.
