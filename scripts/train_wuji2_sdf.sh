#!/usr/bin/env bash
# Full offline random-pose collection -> configuration-conditioned MLP.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${WUJI_SDF_PYTHON:-.venv-contact/bin/python}"
DATA="${WUJI_SDF_DATA:-output/wuji2_sdf/random100k_v1}"
MODEL="${WUJI_SDF_MODEL:-output/wuji2_sdf/mlp100k_v1}"
DEVICE="${WUJI_SDF_DEVICE:-cuda:0}"
"$PYTHON" -u -m retarget.learning.wuji2_sdf sample \
  --output "$DATA" --poses "${WUJI_SDF_POSES:-100000}" --per-link 8 \
  --shard-poses 1024 --pose-batch 32 --device "$DEVICE" --resume
"$PYTHON" -u -m retarget.learning.wuji2_sdf train \
  --data "$DATA" --output "$MODEL" --device "$DEVICE" --resume \
  --epochs "${WUJI_SDF_EPOCHS:-30}" --batch-size "${WUJI_SDF_BATCH:-4096}"
