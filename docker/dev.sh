#!/usr/bin/env bash
# Development, inside the container: the API with live reload, plus Vite's dev server, which hot-reloads the screen and
# forwards /api to the API. Open http://<this machine>:5173.
set -euo pipefail
cd /workspace
mkdir -p "$HOME" "$ANVIL_DATA"
[ -d web/node_modules ] || npm --prefix web ci
trap 'kill 0' EXIT
python -m uvicorn anvil.main:app --app-dir server --host 127.0.0.1 --port 8000 --reload --reload-dir server &
npm --prefix web run dev -- --host 0.0.0.0 --port 5173 --strictPort
