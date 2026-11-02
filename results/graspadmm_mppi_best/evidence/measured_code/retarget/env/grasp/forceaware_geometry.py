"""Contained source spheres for MuJoCo's per-geom convex mesh collision bodies.

This does not claim equality to an original, possibly concave triangle mesh.
Every input sample is retained and matched by physical mesh/body provenance;
the ContactAware material points and their indices are never changed.
"""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError


RADIUS_FLOOR_M = 1e-9
MAX_INWARD_INSET_M = 1e-6
SOURCE_OUTSIDE_TOLERANCE_M = 1e-7
CONTAINMENT_TOLERANCE_M = 0.
PARALLEL_DENOMINATOR_EPSILON = 1e-12


def _owner_id(model, owner):
    value = model.body(owner).id if isinstance(owner, str) else int(owner)
    if not 0 <= value < model.nbody:
        raise ValueError("Invalid source-sphere owner body")
    return value


def _rigid_descendant(model, owner, body):
    while body != owner and body != 0:
        if model.body_jntnum[body] != 0:
            return False
        body = int(model.body_parentid[body])
    return body == owner


def _physical_hulls(model, owner, geom_ids):
    """Keep the exact compiled mesh placement, expressed in the owner frame."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rotation = data.xmat[owner].reshape(3, 3)
    groups = {}
    for geom in sorted(set(map(int, geom_ids))):
        if not 0 <= geom < model.ngeom:
            raise ValueError("Physical source geom index is outside the model")
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Contained source spheres require physical mesh geoms")
        body = int(model.geom_bodyid[geom])
        if not _rigid_descendant(model, owner, body):
            raise ValueError("Physical source geom is not rigidly attached to its owner")
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            raise ValueError("A visual-only geom cannot become a physical source sphere")
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start + count].astype(np.float64)
        vertices = (vertices @ data.geom_xmat[geom].reshape(3, 3).T
                    + data.geom_xpos[geom] - data.xpos[owner]) @ rotation
        try:
            hull = ConvexHull(vertices)
        except QhullError as error:
            raise ValueError(f"Physical mesh {model.mesh(mesh).name!r} has no 3-D convex hull") from error
        planes = np.unique(hull.equations, axis=0)
        planes = planes / np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        centroid = vertices[hull.vertices].mean(axis=0)
        if np.any(centroid @ planes[:, :3].T + planes[:, 3] >= 0.):
            raise ValueError("Physical mesh hull has no strictly interior centroid")
        key = (model.mesh(mesh).name, model.body(body).name)
        # Keep all physical candidates for repeated mesh/body names. A sample
        # may match one of several placements; never substitute a visual geom.
        groups.setdefault(key, []).append(dict(geom=geom, planes=planes,
            centroid=centroid, vertices=vertices))
    return groups


def _round_radius_down(values):
    result = np.asarray(values, np.float32)
    above = result.astype(np.float64) > values
    result[above] = np.nextafter(result[above], np.float32(-np.inf))
    return result


def _radius_bound(slack, denominator, maximum, *, clearance=0.):
    with np.errstate(divide="ignore", invalid="ignore"):
        limits = np.where(denominator > PARALLEL_DENOMINATOR_EPSILON,
                          (-slack - clearance) / denominator, np.inf)
    return np.minimum(float(maximum), np.min(limits, axis=1))


def _serialized_fit(points, normals, tangent, radius):
    radii = _round_radius_down(np.maximum(radius, 0.))
    centers = (tangent - radii.astype(np.float64)[:, None] * normals).astype(np.float32)
    displacement = np.linalg.norm(centers.astype(np.float64)
        + radii.astype(np.float64)[:, None] * normals - points, axis=1)
    return centers, radii, displacement


def _fit_to_hull(points, normals, hull, maximum):
    planes = hull["planes"]
    a, b = planes[:, :3], planes[:, 3]
    slack = points @ a.T + b
    denominator = np.maximum(1. - normals @ a.T, 0.)
    # Choose a representable positive floor, rather than accidentally returning
    # the float32 value immediately below the advertised floor.
    floor = float(np.nextafter(np.float32(RADIUS_FLOOR_M), np.float32(np.inf)))
    radius = _radius_bound(slack, denominator, maximum)
    centers, radii, displacement = _serialized_fit(points, normals, points, radius)
    residual = centers.astype(np.float64) @ a.T + b + radii.astype(np.float64)[:, None]
    source_near_hull = np.max(slack, axis=1) <= SOURCE_OUTSIDE_TOLERANCE_M
    valid = source_near_hull & (radii.astype(np.float64) >= floor)
    valid &= np.max(residual, axis=1) <= CONTAINMENT_TOLERANCE_M
    inset = np.zeros(len(points), np.float64)
    retry = np.flatnonzero(source_near_hull & ~valid)
    if len(retry):
        p, n, s, den = points[retry], normals[retry], slack[retry], denominator[retry]
        centroid = hull["centroid"]
        center_slack = centroid @ a.T + b
        decrease = s - center_slack
        # A small interior guard covers the final float32 center/radius
        # serialization. It moves only spheres needing a floor/containment
        # repair; the original CA samples remain untouched.
        guard = max(2e-9, float(np.max(np.abs(hull["vertices"])))
                    * 4. * np.finfo(np.float32).eps)
        required = s + 2. * floor * den + guard
        impossible = np.any((required > 0.) & (decrease <= 0.), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = np.where(required > 0., required / decrease, 0.)
        fraction = np.max(fractions, axis=1)
        tangent = p + fraction[:, None] * (centroid - p)
        local_inset = np.linalg.norm(tangent - p, axis=1)
        # Half the inset guard remains after radius selection. Subtracting the
        # entire guard here would amplify ~1e-18 cancellation by an almost
        # parallel denominator, spuriously rejecting a valid positive sphere.
        new_radius = _radius_bound(tangent @ a.T + b, den, maximum, clearance=.5 * guard)
        c, r, offset = _serialized_fit(p, n, tangent, new_radius)
        error = c.astype(np.float64) @ a.T + b + r.astype(np.float64)[:, None]
        good = ~impossible & np.isfinite(fraction) & (fraction >= 0.) & (fraction <= 1.)
        good &= (local_inset <= MAX_INWARD_INSET_M) & (offset <= MAX_INWARD_INSET_M)
        good &= r.astype(np.float64) >= floor
        good &= np.max(error, axis=1) <= CONTAINMENT_TOLERANCE_M
        chosen = retry[good]
        centers[chosen], radii[chosen] = c[good], r[good]
        displacement[chosen], inset[chosen] = offset[good], local_inset[good]
        valid[chosen] = True
    return dict(centers=centers, radii=radii, valid=valid, inset=inset,
                displacement=displacement, source_slack=np.max(slack, axis=1))


def constrain_tangent_spheres(model, owner, geom_ids, surface, normals,
                              source_meshes, source_bodies, max_radius):
    """Return ``centers, radii, report`` with every sample physically matched.

    For normalized hull halfspaces ``a.x+b <= 0`` and normal ``n``, a sphere
    centered at ``p-r*n`` is contained iff ``slack+r*(1-a.n) <= 0``. Thus each
    nonparallel plane bounds r by ``-slack/(1-a.n)``. At a numerical corner the
    sphere's tangent point may move at most 1 micrometer toward the hull's
    interior centroid to permit a positive radius. The CA point is not moved.

    Results are checked after float32 serialization. The certificate is for
    the corresponding MuJoCo per-geom convex collision body, not exact
    containment in a nonconvex triangle mesh or coverage of the whole mesh.
    """
    if isinstance(max_radius, (bool, np.bool_)) or not np.isscalar(max_radius):
        raise ValueError("Maximum source-sphere radius must be a finite scalar")
    maximum = float(max_radius)
    floor = float(np.nextafter(np.float32(RADIUS_FLOOR_M), np.float32(np.inf)))
    if not np.isfinite(maximum) or maximum < floor:
        raise ValueError("Maximum source-sphere radius must accommodate the positive numerical floor")
    points = np.asarray(surface, np.float64)
    directions = np.asarray(normals, np.float64)
    meshes, bodies = np.asarray(source_meshes).astype(str), np.asarray(source_bodies).astype(str)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0
            or directions.shape != points.shape or meshes.shape != (len(points),)
            or bodies.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(directions).all()):
        raise ValueError("Every finite surface sample needs a normal and mesh/body provenance")
    length = np.linalg.norm(directions, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("Source-sphere normals must be nonzero")
    directions = directions / length[:, None]
    owner = _owner_id(model, owner)
    hulls = _physical_hulls(model, owner, geom_ids)
    keys = sorted(set(zip(meshes, bodies)))
    missing = [key for key in keys if key not in hulls]
    if missing:
        raise ValueError(f"Query provenance has no matching physical geom under {model.body(owner).name}: {missing}")
    centers = np.empty(points.shape, np.float32)
    radii = np.full(len(points), -1., np.float32)
    selected = np.full(len(points), -1, np.int32)
    insets, displacements = np.zeros(len(points)), np.zeros(len(points))
    components = []
    maximum_residual = -np.inf
    for mesh, body in keys:
        ids = np.flatnonzero((meshes == mesh) & (bodies == body))
        source_slack = []
        for candidate in hulls[(mesh, body)]:
            fit = _fit_to_hull(points[ids], directions[ids], candidate, maximum)
            source_slack.append(fit["source_slack"])
            choose = fit["valid"] & (fit["radii"] > radii[ids])
            chosen = ids[choose]
            centers[chosen], radii[chosen] = fit["centers"][choose], fit["radii"][choose]
            selected[chosen] = candidate["geom"]
            insets[chosen], displacements[chosen] = fit["inset"][choose], fit["displacement"][choose]
        if np.any(selected[ids] < 0):
            bad = ids[selected[ids] < 0]
            nearest_slack = np.min(np.stack(source_slack), axis=0)[selected[ids] < 0]
            raise ValueError("Cannot contain positive source spheres without exceeding the allowed inward inset: "
                f"mesh={mesh}, body={body}, sample_ids={bad[:8].tolist()}, "
                f"minimum_candidate_hull_excess_m={nearest_slack[:8].tolist()}")
        for candidate in hulls[(mesh, body)]:
            chosen = ids[selected[ids] == candidate["geom"]]
            if not len(chosen):
                continue
            planes = candidate["planes"]
            residual = (centers[chosen].astype(np.float64) @ planes[:, :3].T
                        + planes[:, 3] + radii[chosen].astype(np.float64)[:, None])
            excess = float(np.max(residual))
            maximum_residual = max(maximum_residual, excess)
            if excess > CONTAINMENT_TOLERANCE_M:
                raise ValueError("Serialized source sphere escaped its physical convex hull")
            components.append(dict(source_mesh=mesh, source_body=body, physical_geom_id=candidate["geom"],
                samples=int(len(chosen)), hull_plane_count=int(len(planes)),
                radius_min_m=float(radii[chosen].min()), radius_max_m=float(radii[chosen].max()),
                inset_samples=int(np.count_nonzero(insets[chosen])),
                max_inward_inset_m=float(insets[chosen].max()), max_halfspace_excess_m=excess))
    report = dict(method="per_physical_geom_convex_hull_contained_tangent_spheres_v1",
        owner_body=model.body(owner).name, owner_body_id=owner,
        input_samples=int(len(points)), output_samples=int(len(points)),
        semantic_filter_applied=False, physical_provenance_filter_applied=False,
        unmatched_provenance_policy="fail closed; never enable a visual-only collision mesh",
        representation="Contained in the corresponding MuJoCo per-geom convex collision body; not exact nonconvex triangle-mesh containment or full surface coverage",
        radius_requested_max_m=maximum, radius_numerical_floor_m=RADIUS_FLOOR_M,
        radius_min_m=float(radii.min()), radius_max_m=float(radii.max()),
        radius_quantiles_m=np.quantile(radii.astype(np.float64), [0., .1, .5, .9, 1.]).tolist(),
        radius_reduced_samples=int(np.count_nonzero(radii.astype(np.float64) < maximum - 1e-9)),
        allowed_max_inward_inset_m=MAX_INWARD_INSET_M, inset_samples=int(np.count_nonzero(insets)),
        max_inward_inset_m=float(insets.max()), max_effective_tangent_point_displacement_m=float(displacements.max()),
        ca_query_points_modified=False, ca_query_indices_modified=False,
        source_outside_tolerance_m=SOURCE_OUTSIDE_TOLERANCE_M,
        containment_tolerance_m=CONTAINMENT_TOLERANCE_M,
        serialized_float32_halfspaces_verified=True, max_halfspace_excess_m=maximum_residual,
        selected_physical_geom_ids=selected.tolist(), components=components)
    return centers, radii, report
