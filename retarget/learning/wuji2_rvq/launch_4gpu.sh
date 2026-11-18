#!/usr/bin/env bash
# Run from the repository root. Pass --resume to continue an existing campaign.
set -euo pipefail
RVQ_PYTHON=${RVQ_PYTHON:-/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
exec "$RVQ_PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m retarget.learning.wuji2_rvq train \
  --output output/wuji2_rvq_joint_4gpu_muon_v1 \
  --task joint --reconstruction-weight 1 --classification-weight 0.05 --optimizer muon --batch-size 8192 --steps-per-epoch 250 --no-compile --threads 2 "$@"
