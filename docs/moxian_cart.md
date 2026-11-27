# Current CART

The maintained trainer is `scripts/train_cart_reach_policy.py`. It uses future ten measured EMF/tactile frames and the previous 20 finger angles plus six wrist coordinates. A shared tactile PointNet pools five finger groups and the palm. Attention combines these tokens with EMF and state; five finger decoders and a wrist decoder predict increments.

Training uses exact differentiable FK and native semantic-palmar contact projection. The objective is normalized CARD positional contact plus native pose smoothness, with a local delta-direction regularizer (weight 0.01). It does not include CARD teacher labels, a learned collision model, or self-collision penalties. This reduced objective is not the full original CARD objective.

Initial and per-step joint noise are zero. Detached rollout depths 0–15 cover recurrent states; only the final step receives gradients, without BPTT. Paired 1–3 mm observation translations remain for the delta-direction term. Finger mechanical limits remain; wrist increments and pose have no hard range limit.

Default width is 384, decoder width 744, effective batch 8192, micro-batch 512, Muon+SGD, and 1000 equivalent epochs. Validation uses noise-free 16-step rollouts on held-out sessions; best checkpoints use the native validation objective.

```sh
python scripts/train_cart_reach_policy.py --root /path/to/cart \
  --reference-config /path/to/reference_config.json --run-id new_run
```

The reference JSON supplies disjoint `train_ids` and `validation_ids`, `seed`, and W&B `tracking` settings. Recordings and robot resources are local dependencies, excluded from Git. Use `MOXIAN_QNBOT_RESOURCES` for robot assets. Web/CLI training selects this same configuration. Existing checkpoint loading utilities retain required tensor-layout compatibility.

See [delta supervision](cart_delta_direction.md) and [devbox operation](devbox_cart_training.md). CARD, GeoRT and the other retargeting algorithms remain independent supported baselines.
