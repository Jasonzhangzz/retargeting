# Devbox CART operation

Canonical source: `/home/jasonzhang/retargeting` (`retargeting-cart` is an alias).
Current run: `devbox_cart_v28_no_noise_20261127`, RTX 5090 GPU 7, fresh initialization, micro-batch 512 × 16 accumulation = 8192. Maximum budget is 1295 updates / 1000 equivalent epochs. It has zero initial and transition joint noise and no CARD teacher supervision.

Run source, launch receipt, logs and checkpoints are isolated under `/home/jasonzhang/cart-runs/<run>/`; cleaning the canonical repository does not mutate active training. Other control runs are also isolated. Their runtime snapshots and results are not repository source and are not uploaded.

W&B: https://wandb.plat.x2robot.com/x2robot/cart/runs/devbox_cart_v28_no_noise_20261127

Runtime: `/tmp/jasonzhang-cart-runtime/venv/bin/python`; recordings are under `/tmp/jasonzhang-cart-runtime/recordings`. These temporary dependencies must be restored after container replacement. Persistent checkpoints are under the run directory. Credentials stay outside Git and launch receipts.

The cleanup backup is `/home/jasonzhang/retargeting-before-cleanup-20261127.tar.gz`, outside the canonical repository. Git history is retained; obsolete learned-geometry training implementations and template duplicates have been removed from the current tree.
