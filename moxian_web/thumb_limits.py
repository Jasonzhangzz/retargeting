"""Virtual thumb range relative to physical straight, not vendor raw zero.

10 degrees is an explicit configurable modelling allowance, not an estimate
of an individual's measured motion range. Original installed assets stay intact.
"""
import math
import xml.etree.ElementTree as ET

THUMB_EXTENSION_DEG=10.
REVISION='thumb_bounded_extension_v2'


def extension_bounds(name,lower,upper,straight):
    if name.endswith(('_thumb_mcp_flexion','_thumb_ip_flexion')):
        return straight-math.radians(THUMB_EXTENSION_DEG),max(upper,straight)
    return lower,upper


def native_thumb_straight(path,side):
    joints={j.get('name'):j for j in ET.parse(path).getroot().findall('joint')}
    def yaw(name):return float(joints[side+'_thumb_'+name].find('origin').get('rpy','0 0 0').split()[2])
    return {side+'_thumb_mcp_flexion':-yaw('mcp_abduction')-yaw('mcp_flexion'),side+'_thumb_ip_flexion':-yaw('ip_flexion')}
