#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ ! -x .venv-moxian-official/bin/python ]]; then
  python3 -m venv .venv-moxian-official
fi
.venv-moxian-official/bin/python -m pip install qnbot-sdk-core==1.8.1 qnbot-sdk-glove==1.8.1
.venv-moxian-official/bin/python -m pip check
.venv-moxian-official/bin/python -I -c 'from qnbot_sdk import Sdk; from qnbot_sdk.glove import ExternalConnection; print("Official SDK 1.8.1 reference ready")'
