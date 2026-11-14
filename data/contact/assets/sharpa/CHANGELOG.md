# Changelog

## Unreleased

### Added

- `left_hand_fit.xml` and `right_hand_fit.xml`: the same hands with every finger
  collision capsule and cylinder (20 per hand) refit to its link's visual
  meshes: end points and radius by least squares on the meshes' support
  function over 2000 directions, sticking out past the mesh weighted 5x, so the
  drawn fingertips touch where the collision geoms do (in a tripod grasp the
  drawn index and middle pads sat 0.9-1.7 mm behind their old capsules). The
  original files are unchanged.
- Initial MJCF model of the Sharpa Wave hand (left and right) derived from the
  upstream URDF at
  [sharpa-robotics/sharpa-urdf-usd-xml@6eea427eb24189519f32b9f21674cd534d3f973c](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/commit/6eea427eb24189519f32b9f21674cd534d3f973c).
- 22 position actuators with joint-class–specific gains.
- VHACD palm collision decomposition (32 pieces) and capsule fits for finger
  phalanges.
- `make_right.py` mirror-generation script and the `meshes_right/` mirrored
  asset directory.
- Names on the fingertip collision geoms of both hands, e.g.
  `right_thumb_fingertip_collision`.

### Fixed

- Right hand joint axes now match the real hardware. `make_right.py` mirrored
  the left hand's motion, which actuated every joint backwards; it now relabels
  the joint coordinate instead (negate the axis, keep the range).
