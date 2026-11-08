#!/usr/bin/env bash
set -euo pipefail

# Reproducible Linux/CI harness. Uses disposable state and never prints secrets.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend" && pwd)"
export UPLOAD_DIR="$(mktemp -d)"
export ARTIFACTS_DIR="$(mktemp -d)"
export STATE_DIR="$(mktemp -d)"
export AI_DB_FILE="$STATE_DIR/ai_control_plane.sqlite3"
export AI_MODEL="${AI_MODEL:-gemini-2.5-flash}"

python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pytest "$ROOT_DIR/tests/test_ml.py" "$ROOT_DIR/tests/test_ai_execution.py" "$ROOT_DIR/tests/test_ai_security_persistence.py" "$ROOT_DIR/tests/test_ai_http_asgi.py" -q

