#!/usr/bin/env bash
# Optional MuJoCo 4 validation environment. The live GPU stack uses requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
UV="${UV:-uv}"
"$UV" venv --python 3.11 .venv-contact4
# Reuse shared dependencies while isolating the benchmark's incompatible engine.
awk '/^# Compute/{exit} {print}' requirements.txt | "$UV" pip install --python .venv-contact4/bin/python -r /dev/stdin
"$UV" pip install --python .venv-contact4/bin/python mujoco==4.0.0 warp-lang==1.16.0 -e ./third_party/robokit
echo 'CPU4 ready; the live GPU runtime remains in .venv-contact.'
