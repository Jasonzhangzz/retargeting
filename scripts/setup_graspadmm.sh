#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [ ! -x .venv-graspadmm/bin/python ]; then
  uv venv --python 3.12 .venv-graspadmm
fi
uv pip install --python .venv-graspadmm/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-graspadmm/bin/python -r requirements-graspadmm.txt
.venv-graspadmm/bin/python -c 'import torch,mujoco,warp,qpsolvers; print("GraspADMM:",torch.__version__,mujoco.__version__,warp.__version__,qpsolvers.available_solvers)'
