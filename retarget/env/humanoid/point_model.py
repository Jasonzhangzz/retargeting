"""SCSP-only native collision model with four contacts at named site centres.

The original robot geoms remain as inertial/visual geometry, with collision
disabled. Four massless spheres of numerical radius 1e-4 m represent the two
palms and two soles. Source XML files and the original actuator/joint model are
unchanged. Object/environment collision pairs keep their original parameters.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def _subtree(model, root):
    result = np.zeros(model.nbody, bool)
    result[root] = True
    for body in range(root+1, model.nbody):
        result[body] = result[int(model.body_parentid[body])]
    return result


def make_point_contact_model(path, hand_sites, foot_sites, *, radius=1e-4, friction=.8):
    """Compile a private model with exactly four robot collision point geoms.

    Palms collide with the object's collision geoms and static ground; soles
    collide with static ground. Native pair solver parameters are copied from
    the closest original geom on the same rigid hand/foot body group. Requested
    ``friction`` sets the two sliding coefficients; rolling/torsional friction
    and solver impedance remain those of the original pair or geom.
    """
    hands, feet = tuple(hand_sites), tuple(foot_sites)
    names = hands + feet
    if len(hands) != 2 or len(feet) != 2 or len(set(names)) != 4 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("point contact model requires two distinct palm and two distinct sole site names")
    if isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("point contact radius must be finite and positive")
    if isinstance(friction, bool) or not math.isfinite(friction) or friction < 0:
        raise ValueError("point contact friction must be finite and nonnegative")
    spec = mujoco.MjSpec.from_file(str(path))
    # Give unnamed input geoms private names before compilation, so deletion of
    # their explicit pairs never relies on compiler-dependent element ordering.
    reserved = {geom.name for geom in spec.geoms if geom.name}
    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            name = f"scsp_original_geom_{i}"
            while name in reserved:
                name += "_"
            geom.name = name
            reserved.add(name)
    model = spec.compile()
    free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free) < 2:
        raise ValueError("point contact model requires separate robot and object free joints")
    robot = _subtree(model, int(model.jnt_bodyid[free[0]]))
    objects = _subtree(model, int(model.jnt_bodyid[free[-1]]))
    if bool((robot & objects).any()):
        raise ValueError("robot and object must be separate body subtrees")
    site_ids = []
    for name in names:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0 or not robot[model.site_bodyid[site]]:
            raise ValueError(f"point contact site {name!r} must belong to the robot")
        if f"scsp_{name}" in reserved:
            raise ValueError(f"point contact geom name scsp_{name!s} already exists")
        site_ids.append(site)
    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    robot_geom = robot[model.geom_bodyid]
    paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
    collision = ((model.geom_contype | model.geom_conaffinity) != 0)
    if paired:
        collision[np.asarray(sorted(paired), int)] = True
    object_geoms = np.flatnonzero(objects[model.geom_bodyid] & collision)
    ground_geoms = np.flatnonzero((model.body_weldid[model.geom_bodyid] == 0) & collision & ~robot_geom & ~objects[model.geom_bodyid])
    if not len(object_geoms):
        raise ValueError("point contact model requires object collision geometry")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    old_pairs = {}
    for pair in list(spec.pairs):
        old_pairs[frozenset((pair.geomname1, pair.geomname2))] = {
            key: np.asarray(getattr(pair, key)).copy() if key not in ("margin", "gap") else float(getattr(pair, key))
            for key in ("solref", "solreffriction", "solimp", "friction", "margin", "gap")
        }
    old_robot_names = {geom_names[i] for i in np.flatnonzero(robot_geom)}
    for geom in spec.geoms:
        if geom.name in old_robot_names:
            geom.contype = 0
            geom.conaffinity = 0
    for pair in list(spec.pairs):
        if pair.geomname1 in old_robot_names or pair.geomname2 in old_robot_names:
            spec.delete(pair)

    for logical, site_id in enumerate(site_ids):
        site_name = names[logical]
        site = spec.site(site_name)
        body_id = int(model.site_bodyid[site_id])
        same_body = model.body_weldid[model.geom_bodyid] == model.body_weldid[body_id]
        sources = np.flatnonzero(robot_geom & same_body & collision)
        if not len(sources):
            sources = np.flatnonzero(robot_geom & same_body)
        if not len(sources):
            raise ValueError(f"point contact site {site_name!r} has no original rigid body geometry")
        distance = np.linalg.norm(data.geom_xpos[sources] - data.site_xpos[site_id], axis=-1)
        sources = sources[np.argsort(distance, kind="stable")]
        source = int(sources[0])
        point_name = f"scsp_{site_name}"
        site.parent.add_geom(
            name=point_name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=site.pos.copy(), quat=site.quat.copy(), size=[radius, 0., 0.],
            mass=0., density=0., contype=0, conaffinity=0, condim=3,
            friction=[friction, model.geom_friction[source, 1], model.geom_friction[source, 2]],
            solref=model.geom_solref[source].copy(), solimp=model.geom_solimp[source].copy(),
            margin=float(model.geom_margin[source]), gap=float(model.geom_gap[source]),
            rgba=[1., .2, .1, 1.], group=5,
        )
        targets = np.concatenate((object_geoms, ground_geoms)) if logical < 2 else ground_geoms
        for target in targets:
            target_name = geom_names[int(target)]
            parameters = None
            for candidate in sources:
                parameters = old_pairs.get(frozenset((geom_names[int(candidate)], target_name)))
                if parameters is not None:
                    parameters = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in parameters.items()}
                    break
            if parameters is None:
                parameters = dict(solref=model.geom_solref[source].copy(), solimp=model.geom_solimp[source].copy(),
                                  friction=np.array([friction, friction, model.geom_friction[source, 1], model.geom_friction[source, 2], model.geom_friction[source, 2]]),
                                  margin=float(model.geom_margin[source]), gap=float(model.geom_gap[source]))
            parameters["friction"][:2] = friction
            spec.add_pair(name=f"scsp_pair_{site_name}_{target_name}", geomname1=point_name,
                          geomname2=target_name, condim=3, **parameters)
    return spec.compile()
