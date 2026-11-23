#!/usr/bin/env bash
# Harbor Oracle Reference Solution for AgentBreak Bench (CryoForge).
# Executes the public-MCP reference solver over SSE (`CRYOFORGE_MCP_URL`)
# using the bundled `solution/eval/reference_solver.py` (never touches `env_core`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_URL="${CRYOFORGE_MCP_URL:-http://cryoforge-mcp:8090/sse}"
SEED="${CRYOFORGE_SEED:-S01}"

PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" -m eval.reference_solver --mcp-url "${MCP_URL}" --seeds "${SEED}" --log-dir /tmp/harbor_oracle_logs "$@"
