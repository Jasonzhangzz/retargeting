"""Actual MuJoCo-Warp contact forces, group loads and wrench-support targets."""
from __future__ import annotations

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from .dataset import BODY_NAMES

SUPPORT_DIM = 256
AUX_DIM = 6 + 6 + SUPPORT_DIM + 2
Vec5 = wp.types.vector(length=5, dtype=wp.float32)


@wp.kernel
def pack_contacts(nacon: wp.array[int], worlds: wp.array[int], geoms: wp.array[wp.vec2i],
                  pos: wp.array[wp.vec3], frame: wp.array[wp.mat33], dist: wp.array[float],
                  force: wp.array[wp.spatial_vector], friction: wp.array(dtype=Vec5), object_geom: wp.array[int],
                  owner: wp.array[int], physical: wp.array[int],
                  groups: wp.array3d[float], allowed: wp.array2d[float],
                  capacity: int, count: wp.array[int], points: wp.array2d[wp.vec3],
                  normals: wp.array2d[wp.vec3], pressures: wp.array2d[float],
                  accepted: wp.array2d[int], bodies: wp.array2d[int], mus: wp.array2d[float],
                  loads: wp.array2d[float], forbidden: wp.array2d[float],
                  depth: wp.array[float], collision: wp.array[float]):
    row = wp.tid()
    if row >= nacon[0]:
        return
    pair = geoms[row]
    a, b = pair[0], pair[1]
    if a < 0 or b < 0:
        return
    world = worlds[row]
    if owner[a] >= 0 and owner[b] >= 0:
        wp.atomic_max(collision, world, wp.max(-dist[row], 0.0))
    hand = int(-1)
    sign = float(0.0)
    if object_geom[a] != 0 and owner[b] >= 0:
        hand = b
        sign = -1.0
    elif object_geom[b] != 0 and owner[a] >= 0:
        hand = a
        sign = 1.0
    if hand < 0:
        return
    wp.atomic_max(depth, world, wp.max(-dist[row], 0.0))
    pressure = force[row][0]
    body = owner[hand]
    finger = physical[hand]
    if pressure > 0.0 or not wp.isfinite(pressure):
        if allowed[world, finger] == 0.0:
            wp.atomic_add(forbidden, world, finger, pressure)
        for group in range(6):
            if groups[world, group, body] != 0.0:
                wp.atomic_add(loads, world, group, pressure)
    if dist[row] > 0.0002 or pressure <= 0.000001:
        return
    slot = wp.atomic_add(count, world, 1)
    if slot >= capacity:
        return
    points[world, slot] = pos[row]
    normals[world, slot] = sign * frame[row][0]
    pressures[world, slot] = pressure
    bodies[world, slot] = body
    mus[world, slot] = wp.min(friction[row][0], friction[row][1])
    permitted = int(0)
    for group in range(6):
        if groups[world, group, body] != 0.0:
            permitted = 1
    accepted[world, slot] = permitted


def geom_labels(model):
    owner = np.full(model.ngeom, -1, np.int32)
    physical = np.zeros(model.ngeom, np.int32)
    object_geom = np.zeros(model.ngeom, np.int32)
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        name = model.body(body).name.split("/")[-1]
        if name == "grasp_object":
            object_geom[geom] = 1
            continue
        while body > 0 and name not in BODY_NAMES:
            if model.body_jntnum[body] > 0:
                break
            body = int(model.body_parentid[body])
            name = model.body(body).name.split("/")[-1]
        if name in BODY_NAMES:
            owner[geom] = BODY_NAMES.index(name)
            physical[geom] = 5 if name == "right_palm_link" else int(name[len("right_finger")]) - 1
    return owner, physical, object_geom


class ContactMeter:
    def __init__(self, sim, group_masks, allowed, radii, *, capacity=128):
        self.sim, self.device = sim, group_masks.device
        self.capacity = capacity
        self.worlds = len(group_masks)
        owner, physical, obj = geom_labels(sim.mj_model)
        dev = str(self.device)
        self.owner = wp.array(owner, dtype=int, device=dev)
        self.physical = wp.array(physical, dtype=int, device=dev)
        self.object_geom = wp.array(obj, dtype=int, device=dev)
        self.groups = wp.from_torch(group_masks.contiguous())
        self.allowed = wp.from_torch(allowed.contiguous())
        self.count = wp.zeros(self.worlds, dtype=int, device=dev)
        self.points = wp.zeros((self.worlds, capacity), dtype=wp.vec3, device=dev)
        self.normals = wp.zeros_like(self.points)
        self.pressures = wp.zeros((self.worlds, capacity), dtype=float, device=dev)
        self.accepted = wp.zeros((self.worlds, capacity), dtype=int, device=dev)
        self.bodies = wp.zeros_like(self.accepted)
        self.mus = wp.zeros_like(self.pressures)
        self.loads = wp.zeros((self.worlds, 6), dtype=float, device=dev)
        self.forbidden = wp.zeros_like(self.loads)
        self.depth = wp.zeros(self.worlds, dtype=float, device=dev)
        self.collision = wp.zeros_like(self.depth)
        self.contact_ids = wp.array(np.arange(sim.wp_data.naconmax, dtype=np.int32), dtype=int, device=dev)
        self.force = wp.zeros(sim.wp_data.naconmax, dtype=wp.spatial_vector, device=dev)
        from retarget.task.dex.cost import grasp_wrench_directions
        self.directions = torch.as_tensor(grasp_wrench_directions(), device=self.device)
        self.radii = radii
        self.slot_ids = torch.arange(capacity, device=self.device)[None]

    def sample(self, center):
        for value in (self.count, self.points, self.normals, self.pressures, self.accepted,
                      self.bodies, self.mus, self.loads, self.forbidden, self.depth, self.collision, self.force):
            value.zero_()
        d = self.sim.wp_data
        # This upstream helper launches on Warp's current device, rather than
        # inferring it from input arrays. Simulation scopes its own calls, but
        # external contact measurements must do so explicitly as well.
        with wp.ScopedDevice(str(self.device)):
            mjw.contact_force(self.sim.wp_model, d, self.contact_ids, False, self.force)
        wp.launch(pack_contacts, dim=d.naconmax,
                  inputs=[d.nacon, d.contact.worldid, d.contact.geom, d.contact.pos, d.contact.frame,
                          d.contact.dist, self.force, d.contact.friction, self.object_geom, self.owner, self.physical,
                          self.groups, self.allowed, self.capacity, self.count, self.points, self.normals,
                          self.pressures, self.accepted, self.bodies, self.mus, self.loads, self.forbidden,
                          self.depth, self.collision], device=str(self.device))
        points, normal = wp.to_torch(self.points), wp.to_torch(self.normals)
        valid = (self.slot_ids < wp.to_torch(self.count)[:, None]) & (wp.to_torch(self.accepted) > 0)
        basis = torch.nn.functional.one_hot(normal.abs().argmin(-1), 3).to(normal.dtype)
        t1 = torch.nn.functional.normalize(torch.cross(normal, basis, dim=-1), dim=-1)
        t2 = torch.cross(normal, t1, dim=-1)
        # Four rays are a conservative inner approximation of circular friction cones.
        forces = normal[:, :, None] + .6 * torch.stack((t1, t2, -t1, -t2), dim=-2)
        forces = forces / np.sqrt(1 + .6**2)
        arms = (points - center[:, None]) / self.radii[:, None, None]
        torque = torch.cross(arms[:, :, None].expand_as(forces), forces, dim=-1)
        rays = torch.cat((forces, torque), -1).flatten(1, 2)
        mask = valid.repeat_interleave(4, 1)
        support = torch.cat([(rays @ ds.T).masked_fill(~mask[:, :, None], -torch.inf).amax(1).clamp_min(0)
                             for ds in self.directions.split(32)], -1)
        self.valid = (wp.to_torch(self.count) <= self.capacity)
        self.valid &= wp.to_torch(d.nacon)[0] <= d.naconmax
        if hasattr(d, "nefc"):
            self.valid &= wp.to_torch(d.nefc) <= d.njmax
        if hasattr(d, "overflow"):
            self.valid = self.valid & (wp.to_torch(d.overflow) == 0)
        aux = torch.cat((wp.to_torch(self.loads), wp.to_torch(self.forbidden), support,
                         wp.to_torch(self.depth)[:, None] * 1000,
                         (wp.to_torch(self.collision) > .0005).float()[:, None]), -1)
        return aux

    def snapshots(self):
        return {key: wp.to_torch(getattr(self, key)).cpu().numpy().copy()
                for key in ("count", "points", "normals", "pressures", "accepted", "bodies", "mus")}

    def audit(self):
        """Independent NumPy reduction of the SAME native solver rows."""
        d = self.sim.wp_data
        count = min(int(wp.to_torch(d.nacon).cpu()[0]), d.naconmax)
        geoms = wp.to_torch(d.contact.geom).cpu().numpy()[:count]
        worlds = wp.to_torch(d.contact.worldid).cpu().numpy()[:count]
        forces = wp.to_torch(self.force).cpu().numpy()[:count, 0]
        owner = wp.to_torch(self.owner).cpu().numpy()
        physical = wp.to_torch(self.physical).cpu().numpy()
        objects = wp.to_torch(self.object_geom).cpu().numpy()
        groups = wp.to_torch(self.groups).cpu().numpy()
        allowed = wp.to_torch(self.allowed).cpu().numpy()
        expected = np.zeros((self.worlds, 6))
        forbidden = np.zeros_like(expected)
        loaded_rows = 0
        for pair, world, force in zip(geoms, worlds, forces):
            a, b = pair
            if min(a, b) < 0:
                continue
            hand = b if objects[a] and owner[b] >= 0 else a if objects[b] and owner[a] >= 0 else -1
            if hand < 0 or not force > 0:
                continue
            loaded_rows += 1
            expected[world] += groups[world, :, owner[hand]] * force
            if allowed[world, physical[hand]] == 0:
                forbidden[world, physical[hand]] += force
        measured = wp.to_torch(self.loads).cpu().numpy()
        wrong = wp.to_torch(self.forbidden).cpu().numpy()
        np.testing.assert_allclose(measured, expected, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(wrong, forbidden, atol=1e-5, rtol=1e-4)
        return dict(loaded_native_rows=loaded_rows,
                    maximum_load_error_n=float(np.max(np.abs(measured - expected))),
                    maximum_forbidden_error_n=float(np.max(np.abs(wrong - forbidden))))
