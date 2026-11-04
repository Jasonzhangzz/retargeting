#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ ! -x .venv-moxian/bin/python ]]; then
  echo '请先执行 scripts/setup_moxian.sh' >&2
  exit 1
fi
exec .venv-moxian/bin/python -u -m moxian_web.server "$@"
