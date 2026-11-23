# syntax=docker/dockerfile:1
# Multi-stage isolated container build for CryoForge RL (Harbor + OpenCode).
# - Stage `cryoforge-mcp`: runs the authoritative physics & commissioning MCP server on :8090.
# - Stage `agent-sandbox` (default): clean `/workspace` container for the evaluated agent
#   with ZERO access to `env_core/`, `mcp_server/`, `eval/`, or verifier secrets.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "mcp>=1.2.0" \
        "pydantic>=2.7.0" \
        "fastapi>=0.115.0" \
        "uvicorn>=0.30.0" \
        "httpx>=0.27.0" \
        "rich>=13.7.0"


FROM base AS cryoforge-mcp

WORKDIR /opt/cryoforge
COPY pyproject.toml requirements.txt ./
COPY env_core ./env_core
COPY mcp_server ./mcp_server

ENV PYTHONPATH=/opt/cryoforge
RUN mkdir -p /var/log/cryoforge/trajectories

EXPOSE 8090
CMD ["python", "-m", "mcp_server.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8090"]


FROM base AS agent-sandbox

WORKDIR /workspace
RUN printf "CryoForge RL Agent Workspace\nAll plant commissioning state and tools are served via the isolated 'cryoforge' MCP server (CRYOFORGE_MCP_URL=http://cryoforge-mcp:8090/sse).\n" > /workspace/README_SANDBOX.txt

CMD ["sleep", "infinity"]
