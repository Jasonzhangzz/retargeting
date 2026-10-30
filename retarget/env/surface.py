"""Task surface measurements using Warp's built-in mesh query.

These samples feed task costs only. They do not create physics contacts or
constraint rows; all physical collision detection is performed by MJWarp.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

from retarget.env.geometry import ObjectMeshSurface
from retarget.env.grasp.hand import HAND_BODIES, query_path


@dataclass
class HandSamples:
    clouds: dict

    @classmethod
    def from_model(cls, cpu):
        clouds = {}
        for name in HAND_BODIES:
            body = cpu.body(name).id
            with np.load(query_path(cpu, name), allow_pickle=False) as raw:
                p = raw["local_pos"].astype(np.float32)
                radii = raw["radius"] if "radius" in raw else np.zeros(len(p))
            clouds[body] = np.column_stack((p, radii)).astype(np.float32)
        return cls(clouds)


@wp.kernel(enable_backward=False)
def _measure(
    mesh: wp.uint64,
    target: int,
    bodies: wp.array(dtype=int),
    starts: wp.array(dtype=int),
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33),
    distance: wp.array2d(dtype=float),
    contact: wp.array2d(dtype=wp.vec3),
    frame: wp.array2d(dtype=wp.mat33),
    active: wp.array2d(dtype=int),
):
    w, row = wp.tid()
    body = bodies[row]
    origin = positions[w, target]
    rot = rotations[w, target]
    best = 1.0e10
    best_point = wp.vec3(0.0)
    normal = wp.vec3(1.0, 0.0, 0.0)
    for i in range(starts[row], starts[row + 1]):
        world = positions[w, body] + rotations[w, body] @ points[i]
        local = wp.transpose(rot) @ (world - origin)
        query = wp.mesh_query_point(mesh, local, 1.0e6)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            delta = local - closest
            gap = query.sign * wp.length(delta) - radii[i]
            if gap < best:
                best = gap
                best_point = origin + rot @ closest
                normal = rot @ (wp.normalize(delta) * query.sign)
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > 0.9:
        axis = wp.vec3(0.0, 1.0, 0.0)
    t1 = wp.normalize(wp.cross(normal, axis))
    t2 = wp.cross(normal, t1)
    distance[w, row] = best
    contact[w, row] = best_point
    frame[w, row] = wp.mat33(
        normal[0], normal[1], normal[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2]
    )
    active[w, row] = int(best <= 0.01)


class SurfaceQueries(HandSamples):
    def __init__(self, cpu, object_body, *, device):
        super().__init__(HandSamples.from_model(cpu).clouds)
        self.device, self.object_body = device, object_body
        self.body_ids = np.array(sorted(self.clouds), np.int32)
        self.output_count = len(self.body_ids)
        self.output_body_ids = wp.array(self.body_ids, dtype=int, device=device)
        self.output_contact_rows = wp.array(
            np.arange(self.output_count), dtype=int, device=device
        )
        self.output_thresholds = wp.full(self.output_count, 0.01, device=device)
        offsets = np.r_[0, np.cumsum([len(self.clouds[b]) for b in self.body_ids])]
        samples = np.concatenate([self.clouds[b] for b in self.body_ids])
        self.starts = wp.array(offsets, dtype=int, device=device)
        self.points = wp.array(samples[:, :3], dtype=wp.vec3, device=device)
        self.radii = wp.array(samples[:, 3], device=device)
        surface = ObjectMeshSurface.from_model(cpu, object_body)
        vertices, inverse = np.unique(
            surface.triangles.reshape(-1, 3), axis=0, return_inverse=True
        )
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(inverse, dtype=int, device=device),
        )
        # Radius and guidance projection use the object's actual mesh vertices.
        self.object_spheres = np.column_stack(
            (vertices, np.full(len(vertices), 1.0e-6))
        ).astype(np.float32)
        geoms = np.flatnonzero(
            (cpu.geom_bodyid == object_body)
            & ((cpu.geom_contype | cpu.geom_conaffinity) != 0)
        )
        if len(geoms) != 1:
            raise ValueError(
                "Task surface measurements require one object collision geom"
            )
        geom = geoms[0]
        self.force_sign = np.full(self.output_count, -1.0, np.float32)
        self.includemargin = np.full(
            self.output_count, cpu.geom_margin[geom] - cpu.geom_gap[geom], np.float32
        )
        self.friction = np.full(
            (self.output_count, 2), cpu.geom_friction[geom, 0], np.float32
        )

    def allocate(self, worlds):
        options = {"device": self.device}
        shape = (worlds, self.output_count)
        return SimpleNamespace(
            contact_distance=wp.zeros(shape, **options),
            contact_position=wp.zeros(shape, dtype=wp.vec3, **options),
            contact_frame=wp.zeros(shape, dtype=wp.mat33, **options),
            contact_active=wp.zeros(shape, dtype=int, **options),
        )

    def measure(self, positions, rotations, result):
        wp.launch(
            _measure,
            result.contact_distance.shape,
            inputs=[
                self.mesh.id,
                self.object_body,
                self.output_body_ids,
                self.starts,
                self.points,
                self.radii,
                positions,
                rotations,
            ],
            outputs=[
                result.contact_distance,
                result.contact_position,
                result.contact_frame,
                result.contact_active,
            ],
            device=self.device,
        )
