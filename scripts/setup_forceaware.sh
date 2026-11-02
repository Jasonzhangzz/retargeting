#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv python install 3.10
if [[ ! -x .venv-forceaware/bin/python ]]; then
    uv venv --python 3.10 .venv-forceaware
fi
uv pip install --python .venv-forceaware/bin/python -r requirements-forceaware.txt
.venv-forceaware/bin/python -c 'from retarget.dynamics.dexforge_upstream import verify_upstream, require_environment; print(verify_upstream()["commit"]); print(require_environment())'
