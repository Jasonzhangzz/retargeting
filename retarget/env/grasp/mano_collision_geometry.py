"""MANO rigid hulls for MJWarp contact and self-collision queries."""
import xml.etree.ElementTree as ET
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
import trimesh
from .batched_physics import BatchedPhysics
from .collision import _depths
from retarget.initialization.dexonomy.util.torch_util import torch_matrix_to_quaternion

class ManoCollisionGeometry:
    def __init__(self, decoder, worlds):
        self.device = str(decoder.device)
        self.worlds = worlds
        root = ET.Element("mujoco")
        option = ET.SubElement(root, "option", gravity="0 0 0")
        ET.SubElement(option, "flag", multiccd="disable")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        contact = ET.SubElement(root, "contact")
        rest = decoder.rest.detach().cpu().numpy()
        joints = decoder.joints_rest.detach().cpu().numpy()
        labels = decoder.base.weights.argmax(-1).cpu().numpy()
        for joint in range(16):
            points = rest[labels == joint] - joints[joint]
            hull = trimesh.convex.convex_hull(points)
            ET.SubElement(
                asset,
                "mesh",
                name=f"mano_{joint}",
                vertex=" ".join(map(str, hull.vertices.ravel())),
                face=" ".join(map(str, hull.faces.ravel())),
            )
            body = ET.SubElement(world, "body", name=f"mano_{joint}")
            ET.SubElement(body, "freejoint")
            ET.SubElement(
                body, "geom", type="mesh", mesh=f"mano_{joint}", condim="1", margin="0"
            )
            parent = int(decoder.base.parents[joint])
            if parent >= 0:
                ET.SubElement(
                    contact, "exclude", body1=f"mano_{parent}", body2=f"mano_{joint}"
                )
        self.xml = ET.tostring(root, encoding="unicode")
        self.cpu = mujoco.MjModel.from_xml_string(self.xml)
        self.physics = (
            BatchedPhysics(self.cpu, worlds, self.device, nconmax=256, capture=False)
            if decoder.device.type == "cuda"
            else None
        )
        self.flags = wp.zeros(self.cpu.ngeom, dtype=int, device=self.device)
        self.depth = wp.zeros((worlds, 2), dtype=float, device=self.device)
        self.joint_order = torch.argsort(decoder.ordered_joint_ids)[:16]

    def query(self, joints, frames):
        n = len(joints)
        q = torch.cat(
            (joints[:, self.joint_order], torch_matrix_to_quaternion(frames)), -1
        ).reshape(n, -1)
        if self.physics is None:
            depth = []
            for value in q.detach().cpu().numpy():
                data = mujoco.MjData(self.cpu)
                data.qpos[:] = value
                mujoco.mj_kinematics(self.cpu, data)
                mujoco.mj_collision(self.cpu, data)
                depth.append(max([0.0] + [-float(c.dist) for c in data.contact]))
            return joints.new_tensor(depth), torch.zeros(
                n, dtype=torch.bool, device=joints.device
            )
        if n < self.worlds:
            q = torch.cat((q, q[:1].expand(self.worlds - n, -1)))
        wp.to_torch(self.physics.data.qpos).copy_(q)
        self.physics.data.overflow.zero_()
        mjw.kinematics(self.physics.model, self.physics.data)
        mjw.collision(self.physics.model, self.physics.data)
        self.depth.zero_()
        wp.launch(
            _depths,
            self.physics.data.naconmax,
            inputs=[
                self.physics.data.nacon,
                self.physics.data.contact.worldid,
                self.physics.data.contact.geom,
                self.physics.data.contact.dist,
                self.flags,
                0.0,
                self.depth,
            ],
            device=self.device,
        )
        return wp.to_torch(self.depth)[:n, 1].clone(), (
            wp.to_torch(self.physics.data.overflow)[:n] != 0
        )
