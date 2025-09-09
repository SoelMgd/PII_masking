FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user pyproject.toml ./

RUN uv lock && uv sync --no-install-project

COPY --chown=user src/ src/
COPY --chown=user . .

RUN mkdir -p /home/user/.cache/huggingface

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uv", "run", "app.py"] 