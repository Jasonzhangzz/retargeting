"""Closed, surface-carved solids and uniform conditional volume sampling."""
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class Shape:
    mesh: trimesh.Trimesh
    kind: str
    radius: float
    height: float
    requested_fraction: float
    removed_fraction: float
    cutters: np.ndarray
    surface_seed: np.ndarray


def primitive(kind, radius, height):
    if not np.isfinite([radius, height]).all() or min(radius, height) <= 0:
        raise ValueError('Primitive dimensions must be finite and positive')
    if kind == 'sphere':
        mesh=trimesh.creation.icosphere(subdivisions=3, radius=1.)
        mesh.apply_scale(radius)
        return mesh
    if kind in ('disk', 'cylinder'):
        # Build before scaling: Trimesh's absolute degenerate-triangle tolerance
        # otherwise deletes the cap faces of the 0.05mm-diameter pinch cylinder.
        mesh=trimesh.creation.cylinder(radius=1.,height=height/radius,sections=64)
        mesh.apply_scale(radius)
        return mesh
    if kind in ('plate', 'cuboid'):
        return trimesh.creation.box(extents=[2*radius, 1.4*radius, height])
    raise ValueError(f'Unknown primitive {kind}')


def surface_point(mesh, rng):
    i = rng.choice(len(mesh.faces), p=mesh.area_faces/mesh.area)
    u, v = rng.random(2)
    if u+v > 1:
        u, v = 1-u, 1-v
    return np.array([1-u-v, u, v]) @ mesh.triangles[i], mesh.face_normals[i]


def carve(kind, radius, height, rng, fraction=None):
    """Grow a boundary-anchored, connected ball union; never remove a core.

    All Boolean operations use unit-scale geometry, including micrometre-size
    pinch objects. Only a single connected, closed remainder is admissible.
    A connected growing union intersecting a convex primitive has no enclosed
    cavity; the actual removed intersection is additionally checked explicitly.
    """
    target = float(rng.uniform(0, .5) if fraction is None else fraction)
    if not np.isfinite(target) or not 0 <= target <= .5:
        raise ValueError('Mask fraction must be in [0, .5]')
    base = primitive(kind, radius, height)
    anchor, normal = surface_point(base, rng)
    if target == 0:
        return Shape(base, kind, radius, height, 0., 0., np.empty((0,4)), anchor)
    scale = max(radius, height/2)
    unit = base.copy(); unit.apply_scale(1/scale)
    point = anchor/scale
    # All centres are on the same original boundary neighbourhood. Each new
    # ball overlaps its predecessor; growth starts at the exterior seed.
    tangent = rng.normal(size=3); tangent -= normal*np.dot(tangent, normal)
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    # On very thin/long solids a boundary-centred ball severs the complete
    # cross-section before reaching a useful mask fraction. Start a larger
    # exterior ball tangent to the seed instead; its inward growth shaves the
    # exposed side while preserving the residual cross-section.
    extent=base.extents
    offset=2. if max(extent)/min(extent)>4 else 0.
    centre=point+normal*offset
    centres = np.stack([centre, centre + tangent*rng.uniform(.03,.15)])
    ratios = np.array([1., rng.uniform(.65,.9)])
    def cut(size):
        balls = []
        for c, ratio in zip(centres, ratios):
            b = trimesh.creation.icosphere(subdivisions=2, radius=size*ratio)
            b.apply_translation(c); balls.append(b)
        tool = trimesh.boolean.union(balls, engine='manifold')
        return trimesh.boolean.difference([unit, tool], engine='manifold'), tool
    low, high = 0., 8.
    best = None
    for _ in range(19):
        mid = (low+high)/2
        remainder, tool = cut(mid)
        actual = 1 - (abs(remainder.volume) if len(remainder.faces) else 0.)/abs(unit.volume)
        if actual <= target:
            low = mid; best = (remainder, tool, mid, actual)
        else:
            high = mid
    if best is None:
        raise ValueError('No admissible exterior mask')
    remainder, tool, size, actual = best
    removed = trimesh.boolean.intersection([unit, tool], engine='manifold')
    if (not remainder.is_watertight or not remainder.is_winding_consistent
            or len(remainder.split()) != 1 or len(removed.split()) != 1
            or not 0 <= actual <= .5 or abs(actual-target) > .002):
        raise ValueError('Disconnected, open or inaccurate carved solid')
    remainder.apply_scale(scale)
    return Shape(remainder, kind, radius, height, target, actual,
                 np.c_[centres*scale, ratios*size*scale], anchor)


def primitive_volume_points(kind, radius, height, count, rng):
    if kind == 'sphere':
        direction = rng.normal(size=(count,3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return direction*(radius*np.cbrt(rng.random(count)))[:,None]
    if kind in ('cylinder', 'disk'):
        angle = rng.uniform(0, 2*np.pi, count)
        radial = radius*np.sqrt(rng.random(count))
        return np.c_[radial*np.cos(angle), radial*np.sin(angle),
                     rng.uniform(-height/2,height/2,count)]
    if kind in ('plate', 'cuboid'):
        return rng.uniform(-.5,.5,(count,3))*[2*radius,1.4*radius,height]
    raise ValueError(kind)


def volume_points(shape, count, rng, contains=None):
    """IID uniform over the *saved mesh*; callbacks may perform GPU winding."""
    if count <= 0:
        raise ValueError('Point count must be positive')
    contains = contains or shape.mesh.contains
    chunks = []; remaining = count
    for _ in range(100):
        proposals = primitive_volume_points(shape.kind,shape.radius,shape.height,
                                            max(64, int(remaining*2.2)),rng)
        good = proposals[np.asarray(contains(proposals),bool)][:remaining]
        chunks.append(good); remaining -= len(good)
        if remaining == 0:
            return np.concatenate(chunks).astype(np.float32)
    raise ValueError('Uniform volume rejection sampling stalled')
