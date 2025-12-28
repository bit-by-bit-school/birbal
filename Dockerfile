FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Copy dependency graph
COPY pyproject.toml uv.lock ./


# Install deps
RUN uv pip compile pyproject.toml -o requirements.txt
RUN uv pip sync --system requirements.txt

COPY . .
