"""Axially center Type 6/7/8 cylinder contacts throughout MPPI sampling."""
import numpy as np
import torch
from .template import WujiContactKinematics
from .template_mppi import TemplatePoseObjective


class CenteredCylinderObjective(TemplatePoseObjective):
    def __init__(self, *args, warm_q, **kwargs):
        super().__init__(*args, **kwargs)
        self.warm_q = np.asarray(warm_q).copy()

    def configure_object_regions(self, name, primitive):
        if int(name.partition('_')[0]) not in (6, 7, 8) or primitive['kind'] != 'cylinder':
            raise ValueError('Centered cylinder objective is restricted to Types 6/7/8')
        super().configure_object_regions(name, primitive)
        self.center_fk = WujiContactKinematics(self.cpu, self.bodies.cpu().numpy(), self.device)
        self.finger_slots = [i for i, body in enumerate(self.template.body_names)
                             if body.startswith('right_finger')]
        self.axial_target = float(self.center[2])

    def project(self, q):
        result = super().project(q).clone()
        flat = result.reshape(-1, 26)
        positions, rotations = torch.vmap(self.center_fk.frames)(flat.double())
        points = positions + torch.einsum('bcij,cj->bci', rotations, self.local.double())
        offset = points[:, self.finger_slots, 2].mean(-1) - self.axial_target
        flat[:, 2] -= offset.to(flat.dtype)
        if bool(((flat[:, 2] < self.low[2]) | (flat[:, 2] > self.high[2])).any()):
            raise ValueError('Axial centering exceeds physical root bounds')
        return result

    def optimize(self, initial_q, solver, seed):
        initial = self.project(torch.as_tensor(self.warm_q, device=self.device, dtype=torch.float32))
        self.center_initial_q = initial.cpu().numpy().copy()
        return super().optimize(self.center_initial_q, solver, seed)

    def diagnostics(self, q):
        result = super().diagnostics(q)
        points = np.asarray(result['hand_points'])[self.finger_slots]
        error = float(points[:, 2].mean() - self.axial_target)
        result['axial_centering'] = dict(policy='finger_contact_centroid_at_cylinder_midplane_v1',
            target_z_m=self.axial_target, mean_z_m=float(points[:, 2].mean()),
            error_m=error, min_z_m=float(points[:, 2].min()), max_z_m=float(points[:, 2].max()),
            tolerance_m=1.e-6, passed=abs(error) < 1.e-6, finger_slots=self.finger_slots)
        return result
