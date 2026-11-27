# PointNet single-step CART

See [current architecture and objective](moxian_cart.md). The tactile encoder uses shared point features and fixed anatomical mean pooling into six groups, avoiding per-taxel attention. Overlapping measured frames are encoded once per batch. Pooling and FK retain input and parameter gradients.

Training samples detached current-policy rollout depths 0–15. Initial states and transitions have no injected joint noise. The differentiable final step is supervised by contact, native pose smoothness and the paired local delta-direction loss. At depth zero there is no artificial smoothness edge. Validation remains a 16-step closed-loop rollout.
