#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ ! -x .venv-moxian-ik/bin/python ]]; then
  python3 -m venv .venv-moxian-ik
fi
.venv-moxian-ik/bin/python -m pip install qnbot-sdk-core==1.7.0 qnbot-sdk-glove==1.7.0
.venv-moxian-ik/bin/python -m pip check
.venv-moxian-ik/bin/python -I -c 'from qnbot_sdk import Sdk; from qnbot_sdk.glove import GloveConfig; print("Official Skeleton SDK 1.7.0 ready")'
