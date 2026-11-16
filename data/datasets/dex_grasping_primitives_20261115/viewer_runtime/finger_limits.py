"""Shared Moxian IP extension and root torsion limits.

These are model defaults, not an individual's measured joint ranges.
The installed meshes, URDF origins and joint axes remain unchanged.
"""
import math

REVISION='finger_bounded_extension_v2'
DIP_EXTENSION_DEG=20.
PIP_EXTENSION_DEG=10.
ROOT_TWIST_DEG=30.  # Explicit model bound; not a measured anatomical limit.
FINGERS=('index','middle','ring','little')


def is_finger_dip(name):
    return any(name.endswith('_'+finger+'_dip_flexion') for finger in FINGERS)


def straight_angle(rotation):
    # A local Z flexion follows this fixed body origin. Cancel its yaw;
    # retain its tiny fixed roll/pitch rather than editing the original rig.
    return -math.atan2(float(rotation[1][0]),float(rotation[0][0]))


def extension_bounds(name,lower,upper,straight,allowance_deg=DIP_EXTENSION_DEG):
    if not math.isfinite(allowance_deg) or not 0<=allowance_deg<=90:raise ValueError('末节反折范围无效')
    if is_finger_dip(name):return straight-math.radians(allowance_deg),max(upper,straight)
    if any(name.endswith('_'+f+'_pip_flexion') for f in FINGERS):
        return straight-math.radians(PIP_EXTENSION_DEG),max(upper,straight)
    return lower,upper
