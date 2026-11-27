# Direct-geometry CART objective

For T emitted poses, current loss is:

```
L_contact = mean_frames(sum_active_regions(1600 * rho^2 * gap_metres^2))
L_smooth = sum(t=1..T-1)(
    0.1^2 * ||q[t]-q[t-1]||^2
    + 125 * ||1000*(p[t]-p[t-1])||^2
    + 10^2 * ||axis_angle_near(R[t],omega[t-1])-omega[t-1]||^2
) / (T * 1e6)
L_native = L_contact + L_smooth
L_train = L_native + 0.01 * L_delta_direction
```

Rotation uses CARD's nearest equivalent axis-angle chart. Smoothness applies to
emitted poses, without dt division or an invented
pre-window boundary pose. Contact uses these same noise-free states. Source
force weights rho are in[1,1.5]. Contact-gap jitter is absent; its old metric name
is retained as a detached zero. `motion_jitter` is the native pose-smooth sum,
with `smooth_joint`, `smooth_translation`, `smooth_rotation` components.
There is no style, IK-reference, self-collision, or penetration penalty in total.

The nearest semantic-palmar witness is found on the native mesh. Squared gap
uses the exact first derivative through FK almost everywhere via the envelope
rule; feature switching is piecewise differentiable. CEM using this normalized
contact + native smooth objective achieved91.37% of313 active contact points
below1mm on four16-frame windows with good initial seeds. This finite sample
and finite search budget do not guarantee policy convergence.


Current single-step training uses one native smooth edge, except at depth zero where no previous command exists. Validation averages over the 16-frame trajectory as above. See [current configuration](moxian_cart.md) and [delta direction](cart_delta_direction.md).
