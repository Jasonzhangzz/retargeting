from contextlib import nullcontext
from collections import OrderedDict
import xml.etree.ElementTree as ET
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
import trimesh
from retarget.env.grasp.pose_objective import (
    TemplatePoseObjective,
    PoseObjectiveSettings,
)
from retarget.env.grasp.projected_qp import ProjectedQPClosure
from retarget.env.grasp.batched_physics import BatchedPhysics
from retarget.env.grasp.collision import _depths
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


class ManoModelAdapter:
    def __init__(self, decoder, fixed_contacts, surfaces, chunk):
        self.decoder, self.fixed, self.surfaces = decoder, fixed_contacts, surfaces
        self.settings = PoseObjectiveSettings()
        self.chunk = chunk
        if (
            not hasattr(decoder, "collision_geometry")
            or decoder.collision_geometry.worlds != chunk
        ):
            decoder.collision_geometry = ManoCollisionGeometry(decoder, chunk)
        self.collision = decoder.collision_geometry
        if hasattr(decoder, "artifact_root"):
            from pathlib import Path
            import os

            path = Path(decoder.artifact_root) / "mano_collision_geometry.xml"
            temporary = path.with_suffix(".tmp." + str(os.getpid()))
            temporary.write_text(self.collision.xml)
            temporary.replace(path)
        constants = {}
        for key in dict.fromkeys(surfaces.object_keys):
            geometry_id, scale = key
            geo = surfaces.geometries[geometry_id]
            vertices = (
                torch.as_tensor(
                    geo["vertices"], device=decoder.device, dtype=torch.float64
                )
                * scale
            )
            faces = torch.as_tensor(
                geo["faces"], device=decoder.device, dtype=torch.long
            )
            triangles = vertices[faces]
            volume = (
                triangles[:, 0] * torch.linalg.cross(triangles[:, 1], triangles[:, 2])
            ).sum(-1) / 6
            center = (volume[:, None] * triangles.sum(1) / 4).sum(0) / volume.sum()
            radius = (vertices - center).norm(dim=-1).max().clamp_min(0.005)
            if not geo["watertight"] or not bool(volume.sum() > 0):
                raise ValueError("Closure requires a closed outward object mesh")
            constants[key] = ProjectedQPClosure(
                self.settings.closure, center=center, radius=radius, object_mass=0.1
            )
        metadata = [constants[key] for key in surfaces.object_keys]
        self.centers = torch.stack([m.center for m in metadata]).to(
            device=decoder.device, dtype=torch.float32
        )
        self.radii = torch.stack([m.radius for m in metadata]).to(
            device=decoder.device, dtype=torch.float32
        )
        self.inverses = torch.stack([m.inverse_mass for m in metadata]).to(
            device=decoder.device, dtype=torch.float32
        )
        if not hasattr(decoder, "shared_scorers"):
            decoder.shared_scorers = OrderedDict()
        signature = (
            chunk,
            tuple(
                (r["shadow_body"], int(r["mano_face_id"]), tuple(r["mano_barycentric"]))
                for r in fixed_contacts.records
            ),
        )
        if signature not in decoder.shared_scorers:
            scorer = TemplatePoseObjective.__new__(TemplatePoseObjective)
            scorer.settings = self.settings
            scorer.closure_solver = metadata[0]
            scorer.antipodal_weight = self.settings.antipodal_weight
            scorer.posture = None
            scorer.joint_reference = torch.zeros(
                decoder.components, device=decoder.device
            )
            from retarget.env.grasp.graspadmm_scene import required_body_groups
            from retarget.task.grasp.dexonomy import wuji_body

            names = [wuji_body(r["shadow_body"]) for r in fixed_contacts.records]
            scorer.required_groups = [
                torch.tensor(
                    [i for i, name in enumerate(names) if name in group],
                    device=decoder.device,
                )
                for group in required_body_groups(names)
            ]
            decoder.shared_scorers[signature] = scorer
            if len(decoder.shared_scorers) > 2:
                decoder.shared_scorers.popitem(last=False)
        decoder.shared_scorers.move_to_end(signature)
        self.scorer = decoder.shared_scorers[signature]
        self.last = {}

    def evaluate(self, variables, rows, collect=False):
        output = variables.new_empty(len(variables))
        collected = {}
        device = str(self.decoder.device)
        stream = (
            wp.ScopedStream(
                wp.stream_from_torch(torch.cuda.current_stream(self.decoder.device))
            )
            if self.decoder.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), wp.ScopedDevice(device), stream:
            for start in range(0, len(variables), self.chunk):
                values = variables[start : start + self.chunk]
                ids = rows[start : start + self.chunk]
                count = len(values)
                if count < self.chunk:
                    values = torch.cat(
                        (values, values[:1].expand(self.chunk - count, -1))
                    )
                    ids = torch.cat((ids, ids[:1].expand(self.chunk - count)))
                vertices, joints, frames = self.decoder(values, full=True)
                hand, normal = self.fixed.evaluate(vertices, full=True)
                projected, outward, gap, projection_faces, _ = self.surfaces.query(
                    hand, ids
                )
                _, _, distance, _, _ = self.surfaces.query(vertices, ids)
                self_depth, overflow = self.collision.query(joints, frames)
                depth = torch.stack(
                    ((-distance).clamp_min(0).max(-1).values, self_depth), -1
                )
                rotation = self.surfaces.rotation[ids].transpose(-1, -2)
                origin = self.surfaces.origin[ids, None]
                world = torch.einsum("bij,bkj->bki", rotation, hand - origin)
                hn = torch.einsum("bij,bkj->bki", rotation, normal)
                projection = torch.einsum("bij,bkj->bki", rotation, projected - origin)
                on = torch.einsum("bij,bkj->bki", rotation, outward)
                self.scorer.closure_solver.set_object_batch(
                    self.centers[ids], self.radii[ids], self.inverses[ids]
                )
                cost, parts = self.scorer.score_geometry(
                    values,
                    world,
                    hn,
                    projection,
                    on,
                    gap,
                    depth,
                    overflow,
                    torch.isfinite(values).all(-1),
                )
                parts["projection_face_ids"] = projection_faces
                parts["penetration_vertex"] = distance.argmin(-1)
                output[start : start + count] = cost[:count]
                if collect:
                    for name in parts:
                        if name not in collected:
                            collected[name] = parts[name].new_empty(
                                (len(variables), *parts[name].shape[1:])
                            )
                        collected[name][start : start + count] = parts[name][:count]
        if collect:
            self.last = collected
        return output
