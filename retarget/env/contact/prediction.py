"""Public-API distance queries and conservative advancement of planned geom poses.

Distances here are geometric. Native speculative contact.dist is a different,
solver-effective quantity and must never be interpreted as penetration.
"""

from dataclasses import dataclass, asdict
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class ContactPrediction:
    geom1: int
    geom2: int
    status: str
    toi: float | None = None
    distance: float | None = None
    witness1: list | None = None
    witness2: list | None = None
    normal: list | None = None
    occluded: bool = False
    iterations: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def distance_query(model: mujoco.MjModel, data: mujoco.MjData,
                   geom1: int, geom2: int, cutoff: float = 1e6) -> tuple:
    if geom1 == geom2 or min(geom1, geom2) < 0 or max(geom1, geom2) >= model.ngeom:
        raise ValueError("Two distinct valid geoms are required")
    witness = np.zeros(6)
    dist = float(mujoco.mj_geomDistance(model, data, geom1, geom2, cutoff, witness))
    if dist >= cutoff:
        return dist, None, None, None
    delta = witness[3:] - witness[:3]
    length = np.linalg.norm(delta)
    if model.geom_type[geom1] == mujoco.mjtGeom.mjGEOM_PLANE:
        normal = data.geom_xmat[geom1].reshape(3, 3)[:, 2].copy()
    elif model.geom_type[geom2] == mujoco.mjtGeom.mjGEOM_PLANE:
        normal = -data.geom_xmat[geom2].reshape(3, 3)[:, 2].copy()
    elif length > 1e-12:
        normal = delta / length * (-1 if dist < 0 else 1)
    else:
        # Coincident witnesses at exact contact contain no normal direction.
        # A fixed world-axis fallback can flip a foot or a fingertip in IK.
        # Recover the distance gradient on isolated geom transforms instead.
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, data)
        epsilon = max(1e-5, float(model.opt.ccd_tolerance) * 10)
        gradient = np.zeros(3)
        original = scratch.geom_xpos[geom1].copy()
        for axis in range(3):
            scratch.geom_xpos[geom1] = original
            scratch.geom_xpos[geom1, axis] += epsilon
            plus = mujoco.mj_geomDistance(model, scratch, geom1, geom2, cutoff, None)
            scratch.geom_xpos[geom1, axis] -= 2 * epsilon
            minus = mujoco.mj_geomDistance(model, scratch, geom1, geom2, cutoff, None)
            gradient[axis] = (plus - minus) / (2 * epsilon)
        normal = -gradient / np.linalg.norm(gradient) if np.linalg.norm(gradient) > 1e-8 else np.array([0., 0., 1.])
    return dist, witness[:3].copy(), witness[3:].copy(), normal


def ray_occluded(model: mujoco.MjModel, data: mujoco.MjData, geom1: int,
                 geom2: int, a: np.ndarray, b: np.ndarray) -> bool:
    delta = b - a
    length = np.linalg.norm(delta)
    if length < 1e-8:
        return False
    direction = delta / length
    body = model.geom_bodyid[geom1]
    for g in range(model.ngeom):
        if g in (geom1, geom2) or model.body_weldid[model.geom_bodyid[g]] == model.body_weldid[body]:
            continue
        if not ((model.geom_contype[geom1] & model.geom_conaffinity[g]) or
                (model.geom_contype[g] & model.geom_conaffinity[geom1])):
            continue
        b1, b3 = sorted((int(body), int(model.geom_bodyid[g])))
        if ((b1 << 16) + b3) in model.exclude_signature:
            continue
        origin = a + direction * 1e-6
        typ = int(model.geom_type[g])
        if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
            hit = mujoco.mj_rayMesh(model, data, g, origin, direction)
        elif typ <= int(mujoco.mjtGeom.mjGEOM_BOX):
            hit = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g],
                                     model.geom_size[g], origin, direction, typ)
        else:
            continue
        if 0 <= hit < length - 2e-6:
            return True
    return False


def predict_pair(model: mujoco.MjModel, data: mujoco.MjData, geom1: int,
                 geom2: int, times: np.ndarray, positions: np.ndarray,
                 quaternions: np.ndarray, *, clearance: float = 0.,
                 tolerance: float = 1e-5, max_iterations: int = 32) -> ContactPrediction:
    """First impact on a piecewise linear-position/geodesic-rotation trajectory.

times are seconds relative to the measured state; positions are (T, 2, 3),
quaternions (T, 2, 4), scalar first. Each segment has a conservative rotation
bound. Call again after each dynamics step; this is not acceleration extrapolation.
The caller's MjData and shared model are never changed.
"""
    times, positions, quaternions = map(np.asarray, (times, positions, quaternions))
    if (times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 2, 3)
            or quaternions.shape != (len(times), 2, 4) or np.any(np.diff(times) <= 0)
            or not all(np.isfinite(x).all() for x in (times, positions, quaternions))
            or clearance < 0 or tolerance <= 0 or max_iterations < 1):
        raise ValueError("Invalid planned geom trajectory")
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    gids = [geom1, geom2]
    if geom1 == geom2 or min(gids) < 0 or max(gids) >= model.ngeom:
        raise ValueError("Two distinct valid geoms are required")
    supported = {int(x) for x in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE,
                 mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                 mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX,
                 mujoco.mjtGeom.mjGEOM_MESH)}
    if any(int(model.geom_type[g]) not in supported for g in gids):
        raise ValueError("Prediction requires convex geoms or a static plane")
    iterations = 0
    durations = np.diff(times)
    segments = np.arange(len(durations))
    if all(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE for g in gids):
        # A rotating geom always stays inside its centre's bounding sphere.
        # The closest point on each relative centre segment gives a safe swept
        # broadphase filter before constructing rotations or running GJK.
        relative = positions[:, 1] - positions[:, 0]
        motion = np.diff(relative, axis=0)
        norm2 = np.sum(motion * motion, axis=1)
        fraction = np.clip(-np.sum(relative[:-1] * motion, axis=1) / np.maximum(norm2, 1e-30), 0., 1.)
        closest = relative[:-1] + fraction[:, None] * motion
        radius = float(model.geom_rbound[gids].sum()) + clearance + tolerance
        segments = segments[np.sum(closest * closest, axis=1) <= radius**2]
    for segment in segments:
        duration = durations[segment]
        rotations = [Rotation.from_quat(quaternions[segment:segment + 2, i][:, [1, 2, 3, 0]]) for i in range(2)]
        angular = np.array([(r[0].inv() * r[1]).magnitude() / duration for r in rotations])
        velocity = (positions[segment + 1] - positions[segment]) / duration
        for i, g in enumerate(gids):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and (angular[i] > 1e-12 or np.linalg.norm(velocity[i]) > 1e-12):
                raise ValueError("Moving planes require a finite support geometry")
        sl = [Slerp([0., duration], r) for r in rotations]
        t = 0.
        for _ in range(max_iterations):
            iterations += 1
            for i, g in enumerate(gids):
                scratch.geom_xpos[g] = positions[segment, i] + t * velocity[i]
                scratch.geom_xmat[g] = sl[i]([t]).as_matrix()[0].reshape(9)
            dist, p1, p2, normal = distance_query(model, scratch, geom1, geom2)
            if p1 is None:
                return ContactPrediction(geom1, geom2, 'uncertain', iterations=iterations)
            if dist <= clearance + tolerance:
                occluded = ray_occluded(model, data, geom1, geom2, data.geom_xpos[geom1], p1)
                occluded |= ray_occluded(model, data, geom2, geom1, data.geom_xpos[geom2], p2)
                return ContactPrediction(geom1, geom2, 'occluded' if occluded else 'hit',
                    float(times[segment] + t), dist, p1.tolist(), p2.tolist(), normal.tolist(), bool(occluded), iterations)
            speed = float(normal @ (velocity[0] - velocity[1]) + angular @ model.geom_rbound[gids])
            if speed <= 1e-12:
                break
            advance = (dist - clearance) / speed
            if t + advance > duration:
                break
            t += advance
        else:
            return ContactPrediction(geom1, geom2, 'uncertain', iterations=iterations)
    return ContactPrediction(geom1, geom2, 'no_collision', iterations=iterations)
