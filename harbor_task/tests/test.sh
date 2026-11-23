#!/usr/bin/env bash
# Zero-Trust Harbor Sidecar Verifier for CryoForge RL.
# Queries the isolated `cryoforge-mcp` sidecar container's `/internal/grade`
# endpoint using `X-Verifier-Token` (never exposed to the OpenCode agent container)
# and writes the scalar [0.0, 1.0] reward to `/logs/verifier/reward.txt` plus
# the full authoritative VerificationReport + TrajectoryLog to `/logs/verifier/report.json`.
# Deleting or tampering with `/workspace` in the agent container yields 0.0000 exploit credit.

set -euo pipefail

VERIFIER_DIR="${HARBOR_VERIFIER_DIR:-/logs/verifier}"
GRADE_URL="${CRYOFORGE_GRADE_URL:-http://cryoforge-mcp:8090/internal/grade}"
VERIFIER_TOKEN="${CRYOFORGE_VERIFIER_TOKEN:-cryoforge-harbor-verifier-9f8e7d6c5b4a}"

mkdir -p "${VERIFIER_DIR}"

PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

"${PYTHON_BIN}" - "${GRADE_URL}" "${VERIFIER_TOKEN}" "${VERIFIER_DIR}" <<'PYEOF'
import json
import math
import pathlib
import sys
import urllib.error
import urllib.request

grade_url = sys.argv[1]
verifier_token = sys.argv[2]
verifier_dir = pathlib.Path(sys.argv[3])
reward_path = verifier_dir / "reward.txt"
report_path = verifier_dir / "report.json"


def write_failure(reason: str, details: str = "") -> None:
    reward_path.write_text("0.0000\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "ok": False,
                "final_score": 0.0,
                "error": reason,
                "details": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


try:
    req = urllib.request.Request(
        grade_url,
        headers={
            "Accept": "application/json",
            "X-Verifier-Token": verifier_token,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw_body = resp.read().decode("utf-8")
        payload = json.loads(raw_body)
except urllib.error.HTTPError as exc:
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    write_failure(f"HTTP_{exc.code}", body)
    sys.exit(0)
except Exception as exc:
    write_failure("VERIFIER_REQUEST_FAILED", str(exc))
    sys.exit(0)

if not isinstance(payload, dict) or not payload.get("ok", False):
    write_failure("INVALID_GRADE_RESPONSE", json.dumps(payload))
    sys.exit(0)

raw_score = payload.get("final_score", 0.0)
try:
    score = float(raw_score)
except (TypeError, ValueError):
    score = 0.0

if not math.isfinite(score):
    score = 0.0
score = max(0.0, min(1.0, score))

reward_path.write_text(f"{score:.4f}\n", encoding="utf-8")
report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[cryoforge-verifier] seed={payload.get('seed')} finalized={payload.get('finalized')} reward={score:.4f}")
PYEOF
