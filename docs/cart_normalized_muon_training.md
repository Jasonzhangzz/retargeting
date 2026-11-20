# CARD cost normalization and fresh Muon training

The new run `remote_cardsurface_20261119_norm_muon_v3` uses the complete mean
CARD cost `C` and optimizes `L = C / (C + 1_000_000)`. This maps a nonnegative
cost into `[0, 1)` mathematically. Its derivative is
`1_000_000 / (C + 1_000_000)^2`: all CARD terms share the same multiplier,
preserving their relative weights, the current batch gradient direction, and
the raw objective's minimizers. This transformation is applied after averaging
the complete batch, rather than separately to terms or frames. It does not
claim to correct the contact field's inaccurate geometric derivatives.

`train/total` and `validation/loss` are normalized. `train/raw_cost`,
`validation/raw_cost`, and the existing individual CARD terms retain their raw
units for quality comparisons. A smaller plotted loss alone does not imply
better contact or tracking accuracy. Validation still uses the true mesh and
the original 0.5 mm contact and 5 mm penetration acceptance criteria.

Both policy and online geometry optimizers use PyTorch Muon for dense matrix
parameters and momentum SGD for embedding, bias, and normalization parameters.
No Adam optimizer is used in this run. Momentum is 0.95 with Nesterov enabled;
weight decay is 0.01. Policy Muon uses base LR 0.0005 and
`adjust_lr_fn='match_rms_adamw'` (the name of PyTorch's matrix-size correction,
not an Adam optimizer). Geometry uses base rates 0.00001 for pretrained SDF
parameters and 0.001 for the contact branch. A five-update warmup is followed
by cosine LR decay to 10% of each base rate over the 140 scheduled policy
updates (70 epochs, two real parallel batches per epoch).

Gradient clipping is an emergency global norm cap of 10. Logs include
`gradient_norm_before_clip`, `gradient_norm_after_clip`,
`gradient_clip_multiplier`, `parameter_update_norm`, `parameter_update_rms`,
`parameter_update_max`, and `learning_rate`. Muon normalizes matrix update
directions; reducing the loss scale does not by itself make its parameter
updates smaller. The explicit LR schedule controls the scheduled step size.

The policy, both optimizer states, replay, and training cursor start fresh.
Independent pretrained SDF weights and independently fitted contact-head
weights provide geometry initialization. Old policy checkpoints are archived
and removed from the former run's active checkpoint paths after its process
exits. A `STOP` file prevents the retired controller from restarting.

Implementation: `moxian_web/cart_loss_scaling.py`, `moxian_web/cart_optim.py`,
and the training/geometry/logging integration. Unit checks cover the analytic
normalization derivative, raw-term gradient ratios, bounded values, actual
clipping and update metrics, parameter routing, LR scheduling, and exact
Muon/SGD optimizer-state resume.
