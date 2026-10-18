#!/usr/bin/env bash
# Run anvil for real, inside the container: build the screen once, then serve it and the API on one port.
# Open http://<this machine>:8420.
set -euo pipefail
cd /workspace
mkdir -p "$HOME" "$ANVIL_DATA"
[ -d web/node_modules ] || npm --prefix web ci
npm --prefix web run build
exec python -m uvicorn anvil.main:app --app-dir server --host 0.0.0.0 --port 8420
