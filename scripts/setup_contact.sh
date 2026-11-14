#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UV="${UV:-uv}"
"$UV" venv --python 3.11 .venv-contact4
"$UV" pip install --python .venv-contact4/bin/python -r requirements-contact4.txt
echo 'CPU4 ready; GPU controller uses requirements-contact.txt in a separate environment.'
