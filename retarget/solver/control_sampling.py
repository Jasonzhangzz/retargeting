"""The control-sampling baseline is Robokit's original SimMPC implementation.

ControlPlanner adds native CPU4 review and named-state exchange; it delegates
sampling, interpolation, warm shifts, costs and dynamics to vendored Robokit.
"""


def solve(*args, **kwargs):
    raise TypeError('Use retarget.env.contact.robokit_planner.ControlPlanner with a migrated Robokit Task')
