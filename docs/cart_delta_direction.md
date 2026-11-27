# Local motion direction for the single-step CART policy

The current PointNet trainer uses weight 0.01 and up to 128 paired examples per micro-batch. Contact and native pose-smooth terms are unchanged. This coefficient does not guarantee closed-loop convergence.

For the same detached previous robot state s and measured observation window o,
let y=FK_tip(apply(s, policy(s,o))). Independently translate each finger's EMF
position and corresponding taxels by a random vector of length 1–3 mm, shared
across all ten frames. Palm taxels, timestamps, orientation, validity and contact
flags are unchanged. Invalid finger observations are excluded. This is a local
observation augmentation, not a reconstructed physically feasible glove pose.
The native contact objective is only evaluated on the original measurements.

With y' obtained from the perturbed observation and the identical s:

```
L_delta = mean_valid_fingers(1 - dot(normalize(delta_glove), normalize(y' - y)))
L_train = L_native_contact + L_native_smooth + weight * L_delta
```

Normalization uses epsilon 1e-5 m, as in the vendored GeoRT direction loss.
The constant +1 makes the term nonnegative without changing its gradients.
The input perturbation range is narrower than GeoRT's 1–11 mm. This term matches
direction, not exact displacement magnitude or joint-angle increments. Robot
fingertips and glove translations share the canonical glove-wrist frame; the
floating wrist participates in FK. No collision surrogate or geometric nearest
search is used by this new term. Both prediction branches retain parameter
gradients, but the supplied recurrent state is detached in both.

Paired forwards use FP32 attention to resolve small differences reliably, then
restore the caller's precision. At most128 examples per micro-batch are sampled
for this auxiliary term to bound memory. Native contact still uses all8192
examples/update. With micro-batch512 and accumulation16, direction supervision
uses2048 paired examples/update; these are recorded separately in config.
Native16-step validation and checkpoint selection remain unchanged, so the
training total with this optional term must not be directly compared to the
native validation total. Logs include `native_total`, `delta_direction`,
`delta_direction_weighted`, `delta_direction_cosine`, and `delta_response_mm`.

This regularizer constrains response to observations at fixed previous state.
It does not enforce stability of the recurrent state transition. The v26 audit
found drift even on training windows and static observations. Training on states
visited later in on-policy rollouts remains necessary to address that mismatch;
this can use detached transitions without restoring long BPTT. The current
implementation of the direction term does not by itself change state sampling.

## v27 restart: detached on-policy state coverage

Run `devbox_cart_v27_onpolicy_delta_20261126` starts from scratch on devbox GPU7.
The additional `--on-policy-state-depth 16` cycles warmup depths0..15 over the16
micro-batches of each update (512 ×16 =8192 native examples). Warmup executes the
current policy with the existing finger noise and detaches all of its gradients.
The selected next transition alone is differentiated. At depth0 the native
smoothness term is zero, preserving CARD's first-frame convention; all later
transitions compare the preceding clean command to the current command.
The auxiliary direction pair uses the selected transition's exact observation
offset and previous state, including late rollout states.

Delta direction weight0.01 and128 auxiliary samples/micro-batch give2048 paired
examples/update. Network size, future10 observations, Muon+SGD, state noise,
mechanical finger bounds, free wrist and1295-update budget remain unchanged.
Training total includes the new direction term; native_total and closed-loop
validation retain the original native scale. This is still one-step gradient
training, with extra no-gradient rollout work; v26's0.44s timing does not apply.
Frozen source and launch receipt are under
`/home/jasonzhang/cart-runs/devbox_cart_v27_onpolicy_delta_20261126/`.
W&B: https://wandb.plat.x2robot.com/x2robot/cart/runs/devbox_cart_v27_onpolicy_delta_20261126
