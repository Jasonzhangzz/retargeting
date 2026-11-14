"""Wrapper of yourdfpy.URDF that adds mimic joint writing (consider upstreaming to yourdfpy)."""

from lxml import etree
from yourdfpy import URDF, Joint


class URDFWrapper(URDF):
    def _write_joint(self, xml_parent, joint: Joint):
        xml_element = etree.SubElement(
            xml_parent,
            "joint",
            attrib={
                "name": joint.name,
                "type": joint.type,
            },
        )

        etree.SubElement(xml_element, "parent", attrib={"link": joint.parent})
        etree.SubElement(xml_element, "child", attrib={"link": joint.child})
        self._write_origin(xml_element, joint.origin)
        self._write_axis(xml_element, joint.axis)
        self._write_limit(xml_element, joint.limit)
        self._write_dynamics(xml_element, joint.dynamics)

        # Note: different from the original yourdfpy, we add mimic joint writing here
        if joint.mimic is not None:
            self._write_mimic(xml_element, joint.mimic)
