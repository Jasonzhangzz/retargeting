#!/usr/bin/env bash
# Run alongside an existing job; backfill history and follow future epoch records.
set -euo pipefail
RVQ_PYTHON=${RVQ_PYTHON:-/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/python}
exec "$RVQ_PYTHON" -u -m retarget.learning.wuji2_rvq.wandb_sync \
  --output output/wuji2_rvq_joint_4gpu_muon_v1 \
  --base-url https://wandb.plat.x2robot.com --project wuji2-rvq --wait-for-login "$@"
