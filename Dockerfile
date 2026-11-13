FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/models

# Create target directories for bind mounts
RUN mkdir -p /models /chat-history

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3 \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install official Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

EXPOSE 11434

CMD ["ollama", "serve"]