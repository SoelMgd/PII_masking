FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

# Update system and install required packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user (required for HF Spaces)
RUN useradd -m -u 1000 user

# Switch to user and set environment
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy pyproject.toml and generate lock file
COPY --chown=user pyproject.toml ./

# Generate lock file and install dependencies
RUN uv lock && uv sync --no-install-project

# Copy source code and application files
COPY --chown=user src/ src/
COPY --chown=user . .

# Create necessary directories
RUN mkdir -p /home/user/.cache/huggingface

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI application using uv
CMD ["uv", "run", "app.py"] 