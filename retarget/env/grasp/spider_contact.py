"""Capture-safe native contact pressure for SPIDER contact-retention costs.

Contact pressures come from the current MuJoCo-Warp constraint solution. Like
the native contact buffer, they describe the solver stage before integration;
sampling does not run forward dynamics or update any physical state.
"""

from __future__ import annotations

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .forceaware_validation import FINGER_NAMES, finger_body_labels


def required_finger_mask(cpu, reference_contact_bodies):
    """Map accepted CA bodies to unique physical fingers, including fixed pads."""
    bodies = np.asarray(reference_contact_bodies)
    if (bodies.ndim != 1 or not np.issubdtype(bodies.dtype, np.integer)
            or np.any(bodies < 0) or np.any(bodies >= cpu.nbody)):
        raise ValueError("Reference contact bodies must be a vector of valid integer body IDs")
    labels = finger_body_labels(cpu)[bodies]
    required = np.zeros(5, bool)
    required[labels[labels >= 0]] = True
    return required


@wp.kernel
def _sum_finger_normal_loads(
    nacon: wp.array[int],
    contact_world: wp.array[int],
    contact_geom: wp.array[wp.vec2i],
    contact_efc: wp.array2d[int],
    geom_finger: wp.array[int],
    geom_object: wp.array[int],
    force: wp.array[wp.spatial_vector],
    worlds: int,
    ngeom: int,
    njmax: int,
    loads: wp.array2d[float],
):
    row = wp.tid()
    if row >= nacon[0]:
        return
    world = contact_world[row]
    if world < 0 or world >= worlds:
        return
    pair = contact_geom[row]
    first, second = pair[0], pair[1]
    if first < 0 or first >= ngeom or second < 0 or second >= ngeom:
        return
    address = contact_efc[row, 0]
    if address < 0 or address >= njmax:
        return
    finger = int(-1)
    if geom_object[first] != 0 and geom_object[second] == 0:
        finger = geom_finger[second]
    elif geom_object[second] != 0 and geom_object[first] == 0:
        finger = geom_finger[first]
    if finger < 0 or finger >= 5:
        return
    normal = force[row][0]
    # Preserve nonfinite solver values so the caller can reject invalid costs.
    if normal > 0.0 or not wp.isfinite(normal):
        wp.atomic_add(loads, world, finger, normal)


class ContactLoads:
    """Reusable GPU-only measurements of five-finger normal object loads.

    ``required_fingers`` is a deduplicated bool[5] CA requirement mask. ``sample``
    always returns all five fingers in ``finger_names`` order; its float32
    [worlds,5] buffer is overwritten on the next call. Copy it into a trajectory
    or clone it when a historical sample is needed. Construct/warm the instance
    outside capture, and use the same Torch/Warp stream as the physics caller.
    """

    finger_names = FINGER_NAMES

    def __init__(self, physics, reference_contact_bodies):
        self.physics = physics
        cpu, data = physics.cpu, physics.data
        required = required_finger_mask(cpu, reference_contact_bodies)
        object_body = int(cpu.jnt_bodyid[cpu.joint("obj_joint").id])
        object_bodies = np.zeros(cpu.nbody, np.int32)
        object_bodies[object_body] = 1
        for body in range(object_body+1, cpu.nbody):
            object_bodies[body] = object_bodies[int(cpu.body_parentid[body])]
        geom_finger = finger_body_labels(cpu)[np.asarray(cpu.geom_bodyid)]
        geom_object = object_bodies[np.asarray(cpu.geom_bodyid)]
        self.required_fingers = torch.as_tensor(required, dtype=torch.bool, device=physics.device)
        with wp.ScopedDevice(physics.device):
            self._ids = wp.array(np.arange(data.naconmax, dtype=np.int32), dtype=int)
            self._force = wp.zeros(data.naconmax, dtype=wp.spatial_vector)
            self._loads = wp.zeros((physics.worlds, 5), dtype=float)
            self._geom_finger = wp.array(geom_finger, dtype=int)
            self._geom_object = wp.array(geom_object, dtype=int)
        self._loads_torch = wp.to_torch(self._loads)

    def sample(self):
        """Decode current native contact forces and sum object loads per world.

        All row selection and reduction stay on the device. Inactive capacity
        rows are cleared and ignored, so changing contacts cannot retain a
        previous step's load. This function allocates no tensors and performs
        no host reads, synchronization, or extra physics calls.
        """
        physics, data = self.physics, self.physics.data
        self._force.zero_()
        self._loads.zero_()
        mjw.contact_force(physics.model, data, self._ids, False, self._force)
        wp.launch(_sum_finger_normal_loads, dim=data.naconmax,
            inputs=[data.nacon, data.contact.worldid, data.contact.geom, data.contact.efc_address,
                    self._geom_finger, self._geom_object, self._force,
                    physics.worlds, physics.cpu.ngeom, data.njmax],
            outputs=[self._loads], device=physics.device)
        return self._loads_torch
