"""Contact-specific branches and targets; IK/LM uses the original Robokit stack."""
import mujoco
import numpy as np
from .robokit_ik import ContactIK
from retarget.solver.contact_cem import Branch, ContactEvent, DEFAULTS
from .prediction import distance_query


def project_surface(model: mujoco.MjModel, geom: int, points: np.ndarray) -> tuple:
    points = np.asarray(points).copy()
    size, typ = model.geom_size[geom], model.geom_type[geom]
    normals = np.zeros_like(points)
    if typ == mujoco.mjtGeom.mjGEOM_PLANE:
        points[:, 2] = 0.; normals[:, 2] = 1.
    elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
        normals = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
        zero = np.linalg.norm(normals, axis=1) < .5
        normals[zero] = [0, 0, 1]
        points = normals * size[0]
    elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
        centre = np.zeros_like(points); centre[:, 2] = np.clip(points[:, 2], -size[1], size[1])
        normals = points - centre
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        normals[np.linalg.norm(normals, axis=1) < .5] = [1, 0, 0]
        points = centre + size[0] * normals
    elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
        rho = np.linalg.norm(points[:, :2], axis=1)
        on_cap = np.abs(np.abs(points[:, 2]) - size[1]) < np.abs(rho - size[0])
        points[:, :2] *= np.minimum(rho, size[0])[:, None] / np.maximum(rho[:, None], 1e-12)
        points[:, 2] = np.clip(points[:, 2], -size[1], size[1])
        points[on_cap, 2] = np.where(points[on_cap, 2] >= 0, size[1], -size[1])
        normals[on_cap, 2] = np.where(points[on_cap, 2] >= 0, 1., -1.)
        side = ~on_cap
        normals[side, :2] = points[side, :2] / np.maximum(np.linalg.norm(points[side, :2], axis=1, keepdims=True), 1e-12)
        normals[side & (np.linalg.norm(normals, axis=1) < .5), 0] = 1.
        points[side, :2] = size[0] * normals[side, :2]
    elif typ == mujoco.mjtGeom.mjGEOM_BOX:
        points = np.clip(points, -size, size)
        axes = np.argmin(size - np.abs(points), axis=1)
        sign = np.where(points[np.arange(len(points)), axes] >= 0, 1., -1.)
        points[np.arange(len(points)), axes] = sign * size[axes]
        normals[np.arange(len(points)), axes] = sign
    elif typ == mujoco.mjtGeom.mjGEOM_MESH:
        import trimesh
        meshid = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[meshid]), int(model.mesh_vertnum[meshid])
        hull = trimesh.convex.convex_hull(model.mesh_vert[start:start + count])
        # Native mesh collision uses its convex hull. Nonconvex targets must be
        # supplied as separate convex geoms; never project onto hidden concavities.
        projected, _, triangles = trimesh.proximity.closest_point_naive(hull, points)
        points = projected
        normals = hull.face_normals[triangles]
    else:
        raise ValueError('Contact search targets must be convex primitives or convex mesh parts')
    return points, normals


class ContactStrategy:
    def __init__(self, task, rollout):
        self.task, self.rollout, self.model = task, rollout, task.model
        m = self.model
        if np.any(m.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError('Contact LM requires direct joint position servos')
        self.joints = m.actuator_trnid[:, 0]
        if len(np.unique(self.joints)) != m.nu or np.any(m.jnt_type[self.joints] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Contact LM requires unique actuated hinge joints')
        self.qadr = m.jnt_qposadr[self.joints]
        self.dadr = m.jnt_dofadr[self.joints]
        self.ik = ContactIK(task, str(rollout.device))
        self.last_residual = None

    def branches(self, state: dict, packet: dict) -> list:
        task, m = self.task, self.model
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:] = state['qpos'], state['qvel']
        mujoco.mj_forward(m, d)
        targets = [m.geom('floor').id] if task.name == 'h1_jog' else [m.geom('pen' if task.name == 'sharpa_pen' else 'coin').id]
        packet['target_geoms'] = targets
        pairs = []
        for target in targets:
            for link in range(m.ngeom):
                if link == target or not (m.geom_contype[link] or m.geom_conaffinity[link]):
                    continue
                if task.name == 'h1_jog' and m.geom(link).name not in ('left_foot', 'right_foot'):
                    continue
                if task.name == 'sharpa_coin' and not (m.geom_type[link] == mujoco.mjtGeom.mjGEOM_CAPSULE and task.owner[link] in (1, 2, 3, 4)):
                    continue
                dist, p1, p2, normal = distance_query(m, d, link, target)
                if p1 is None:
                    continue
                pairs.append((dist, link, target, p1, p2, normal))
        pairs.sort(key=lambda x: (x[0], x[1]))
        # Avoid spending every branch on multiple convex pieces of one finger.
        unique, groups = [], set()
        for row in pairs:
            group = task.owner[row[1]] if task.name != 'h1_jog' else row[1]
            if group not in groups:
                unique.append(row); groups.add(group)
        if not unique:
            raise ValueError('No reachable contact geometry candidates')
        rows = unique[:DEFAULTS.top_k]
        events = []
        horizon = task.profile.horizon * packet['dt']
        if 'qpos_ref' in packet:
            ref_data = mujoco.MjData(m)
            self.ref_positions, self.ref_matrices = [], []
            for q in packet['qpos_ref']:
                ref_data.qpos[:] = q
                mujoco.mj_kinematics(m, ref_data)
                self.ref_positions.append(ref_data.geom_xpos.copy())
                self.ref_matrices.append(ref_data.geom_xmat.copy().reshape(m.ngeom, 3, 3))
            self.ref_positions = np.asarray(self.ref_positions)
            self.ref_matrices = np.asarray(self.ref_matrices)
        for _, link, target, p1, p2, normal in rows:
            R1, R2 = d.geom_xmat[link].reshape(3, 3), d.geom_xmat[target].reshape(3, 3)
            start = horizon * .35
            duration, normal_speed = max(horizon * .25, packet['dt']), -.02
            point = R2.T @ (p2 - d.geom_xpos[target])
            if task.name == 'h1_jog':
                foot = m.geom(link).name
                lift = packet['ref'][:, task.adr[foot]]
                # First future stance, or the descending end if the horizon is all swing.
                stance = np.flatnonzero(lift < .005)
                landing_step = int(stance[0]) if len(stance) else len(lift) - 1
                start = float(landing_step * packet['dt'])
                liftoff = np.flatnonzero(lift[landing_step + 1:] >= .005)
                duration = float(max(1, int(liftoff[0]) + 1 if len(liftoff) else len(lift) - landing_step) * packet['dt'])
                if 'qpos_ref' in packet:
                    future = mujoco.MjData(m)
                    landing_step = min(task.profile.horizon, round(start / packet['dt']))
                    future.qpos[:] = packet['qpos_ref'][landing_step]
                    mujoco.mj_kinematics(m, future)
                    mujoco.mj_comPos(m, future)
                    _, p1, p2, normal = distance_query(m, future, link, target)
                    R1 = future.geom_xmat[link].reshape(3, 3)
                    point = R2.T @ (p2 - future.geom_xpos[target])
                    # Local witnesses belong to the demonstrated landing pose.
                    link_origin = future.geom_xpos[link]
                    jac = np.zeros((3, m.nv))
                    mujoco.mj_jac(m, future, jac, None, p1, int(m.geom_bodyid[link]))
                    normal_speed = float((jac @ packet['qvel_ref'][landing_step]) @ (-normal))
                else:
                    point[0] += float(np.mean(packet['ref'][:max(1, round(start / packet['dt'])), task.adr['torso_vel']]) * start)
                    link_origin = d.geom_xpos[link]
            else:
                link_origin = d.geom_xpos[link]
            event = ContactEvent(link, target, m.body(int(m.geom_bodyid[link])).name, m.geom(target).name,
                (R1.T @ (p1 - link_origin)).tolist(), point.tolist(), (R1.T @ normal).tolist(),
                (R2.T @ -normal).tolist(), start, duration, normal.tolist(), normal_velocity=normal_speed)
            if task.name == 'h1_jog':
                event.approach_normal_velocity = min(normal_speed, 0.)
                event.normal_velocity = 0.
            events.append(event.as_dict())
        templates = []
        if task.name == 'h1_jog':
            ordered = sorted(events, key=lambda e: e['start_time'])
            # Change which foot carries each stance interval, not event order.
            # Both schedules keep each foot's own local point/landing region.
            alternate = [dict(e, start_time=other['start_time'], duration=other['duration'])
                         for e, other in zip(ordered, reversed(ordered))]
            templates = [ordered, alternate]
        elif task.name == 'sharpa_coin':
            current = 1 + (0, 1, 2, 3, 2, 1)[task.goal % 6]
            next_owner = 1 + (0, 1, 2, 3, 2, 1)[(task.goal + 1) % 6]
            ordered = sorted(events, key=lambda e: (task.owner[e['geom1']] not in (current, next_owner), task.owner[e['geom1']] != current))
            selected = ordered[:2]
            landing = float(np.clip((.7 - max(task.phase, 0.)) * .6, packet['dt'], horizon))
            # Distinct overlap and early-release handoffs, not just a permutation
            # of identical simultaneous contacts on the same pair of fingers.
            templates = [[dict(selected[0], start_time=0., duration=min(horizon, landing + .05)),
                          dict(selected[1], start_time=max(packet['dt'], landing - .05), duration=horizon * .3)],
                         [dict(selected[0], start_time=0., duration=max(packet['dt'], landing - .05)),
                          dict(selected[1], start_time=landing, duration=horizon * .3)]]
        else:
            templates = [events[:2]]
            if len(events) > 2:
                templates.append([events[2], events[3] if len(events) > 3 else events[0]])
        branches = []
        for i, template in enumerate(templates):
            for mode in (('stick', 'slide') if task.name == 'h1_jog' else ('slide', 'roll')):
                selected = [dict(e, mode='stick' if task.name == 'sharpa_coin' and j == 0 else mode)
                            for j, e in enumerate(template[:DEFAULTS.max_events])]
                # Always the same dimension: padded event variables are ignored.
                sigma = np.tile([.005, .005, .005, horizon * .12, horizon * .1, .1, .1, .05, .05, .05], 3)
                sigma[len(selected) * 10:] = 0.
                for j, e in enumerate(selected):
                    if m.geom_type[e['geom2']] == mujoco.mjtGeom.mjGEOM_PLANE:
                        sigma[j * 10 + 2] = 0.
                    if e['mode'] == 'stick':
                        sigma[j * 10 + 8:j * 10 + 10] = 0.
                branches.append(Branch(selected, np.zeros(30), sigma, f'{i}:{mode}'))
        self.templates = branches[:DEFAULTS.top_k]
        self.state, self.packet = state, packet
        self.zero_controls = {}
        return self.templates

    def decode(self, parameters: np.ndarray, branch: int) -> tuple:
        controls, events = self._decode(parameters, branch)
        if self.task.name == 'h1_jog':
            # Search residual contact changes about a complete whole-body plan.
            # In particular, re-solving an unchanged contact must not inject an
            # IK bias on every MPC step and accumulate it into the warm start.
            if branch not in self.zero_controls:
                base, _ = self._decode(np.zeros((2, parameters.shape[1])), branch)
                self.zero_controls[branch] = base[0]
            controls = np.clip(self.nominal[None] + controls - self.zero_controls[branch][None],
                               *self.task.ctrl_range.T)
            controls[np.all(parameters == 0., axis=1)] = self.nominal
        return controls, events

    def _decode(self, parameters: np.ndarray, branch: int) -> tuple:
        task, m, packet = self.task, self.model, self.packet
        count, H, dt = len(parameters), task.profile.horizon, packet['dt']
        nominal = self.nominal
        controls = np.repeat(nominal[None], count, axis=0)
        events = [[] for _ in range(count)]
        for i, template in enumerate(self.templates[branch].events):
            x = parameters[:, i * 10:(i + 1) * 10]
            points, normals = project_surface(m, template['geom2'], np.asarray(template['object_local_point']) + x[:, :3])
            starts = np.clip(template['start_time'] + x[:, 3], 0., (H - 1) * dt)
            durations = np.clip(template['duration'] + x[:, 4], dt, H * dt - starts)
            dirs = np.repeat(np.asarray(template['approach_direction'])[None], count, axis=0)
            dirs[:, :2] += x[:, 5:7]
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
            vn = (template['normal_velocity'] + np.clip(x[:, 7], -.3, .3) if 'qpos_ref' in packet
                  else np.clip(template['normal_velocity'] + x[:, 7], -.3, .1))
            if task.name == 'h1_jog':
                vn = np.clip(template['approach_normal_velocity'] + x[:, 7], -.3, 0.)
            # Derive world object targets from the task's predicted pose, rather
            # than assuming that the moving pen/coin stays at its current pose.
            target = np.empty((count, 3)); target_normal = np.empty_like(target)
            rotations = np.empty((count, 3, 3))
            for b in range(count):
                e = dict(template, object_local_point=points[b].tolist(), object_local_normal=normals[b].tolist(),
                         start_time=float(starts[b]), duration=float(durations[b]), approach_direction=dirs[b].tolist(),
                         normal_velocity=float(vn[b]), tangential_velocity=[0., 0., 0.])
                if task.name == 'h1_jog':
                    e['normal_velocity'], e['approach_normal_velocity'] = 0., float(vn[b])
                events[b].append(e)
                target[b], target_normal[b], rotations[b] = self.target_pose(e)
            # Surface offsets are object-local; approach/velocity outputs are
            # world vectors. Construct a true tangent basis before converting.
            basis1 = np.repeat([[1., 0., 0.]], count, axis=0)
            basis1 -= normals * np.sum(normals * basis1, axis=1, keepdims=True)
            singular = np.linalg.norm(basis1, axis=1) < 1e-8
            basis1[singular] = [0., 1., 0.]
            basis1 /= np.linalg.norm(basis1, axis=1, keepdims=True)
            basis2 = np.cross(normals, basis1)
            tangent_local = x[:, 8:9] * basis1 + x[:, 9:10] * basis2
            if template['mode'] == 'stick':
                tangent_local[:] = 0.
            tangent = np.einsum('bij,bj->bi', rotations, tangent_local)
            dirs = -target_normal + np.einsum('bij,bj->bi', rotations, x[:, 5:6] * basis1 + x[:, 6:7] * basis2)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
            for b in range(count):
                events[b][-1]['tangential_velocity'] = tangent[b].tolist()
                events[b][-1]['approach_direction'] = dirs[b].tolist()
            correction = self.lm(template['geom1'], np.asarray(template['link_local_point']),
                                 np.asarray(template['link_local_normal']), target, -target_normal,
                                 nominal[np.minimum(np.rint(starts / dt).astype(int), H)],
                                 packet['qpos_ref'][np.minimum(np.rint(starts / dt).astype(int), H)] if 'qpos_ref' in packet else None)
            t = np.arange(H + 1) * dt
            envelope = np.exp(-.5 * ((t[None] - starts[:, None]) / np.maximum(durations[:, None], 2 * dt))**2)
            controls += envelope[..., None] * correction[:, None]
            # A second reachable waypoint encodes approach direction and speed.
            approach_dt = np.minimum(durations, .1)
            velocity = dirs * (-vn[:, None]) + tangent
            before = target - approach_dt[:, None] * velocity
            if 'qpos_ref' in packet:
                approach_dt = np.minimum(approach_dt, starts)
                landing = np.minimum(np.rint(starts / dt).astype(int), H)
                preceding = np.maximum(np.rint((starts - approach_dt) / dt).astype(int), 0)
                geom = template['geom1']
                local = np.asarray(template['link_local_point'])
                reference_landing = self.ref_positions[landing, geom] + np.einsum('bij,j->bi', self.ref_matrices[landing, geom], local)
                reference_before = self.ref_positions[preceding, geom] + np.einsum('bij,j->bi', self.ref_matrices[preceding, geom], local)
                nominal_velocity = target_normal * (template['approach_normal_velocity']
                    if task.name == 'h1_jog' else template['normal_velocity'])
                before = reference_before + (target - reference_landing) - approach_dt[:, None] * (velocity - nominal_velocity)
            approach_correction = self.lm(template['geom1'], np.asarray(template['link_local_point']),
                np.asarray(template['link_local_normal']), before, -target_normal,
                nominal[np.maximum(np.rint((starts - approach_dt) / dt).astype(int), 0)],
                packet['qpos_ref'][np.maximum(np.rint((starts - approach_dt) / dt).astype(int), 0)] if 'qpos_ref' in packet else None)
            before_weight = np.clip((starts[:, None] - t[None]) / np.maximum(approach_dt[:, None], dt), 0, 1)
            controls += envelope[..., None] * before_weight[..., None] * (approach_correction - correction)[:, None]
            if template['mode'] == 'roll':
                # Roll along the target surface: follow its changing surface
                # normal while enforcing zero tangential contact slip in cost.
                moved, moved_normals = project_surface(m, template['geom2'], points + tangent_local * durations[:, None])
                end_target, end_normal = [], []
                for b in range(count):
                    end_event = dict(events[b][-1], object_local_point=moved[b].tolist(),
                        object_local_normal=moved_normals[b].tolist(), start_time=min(H * dt, starts[b] + durations[b]))
                    ep, en, _ = self.target_pose(end_event)
                    end_target.append(ep); end_normal.append(-en)
                end_correction = self.lm(template['geom1'], np.asarray(template['link_local_point']),
                    np.asarray(template['link_local_normal']), np.asarray(end_target), np.asarray(end_normal),
                    nominal[np.minimum(np.rint((starts + durations) / dt).astype(int), H)],
                    packet['qpos_ref'][np.minimum(np.rint((starts + durations) / dt).astype(int), H)] if 'qpos_ref' in packet else None)
                after = np.clip((t[None] - starts[:, None]) / durations[:, None], 0, 1)
                controls += envelope[..., None] * after[..., None] * (end_correction - correction)[:, None]
        controls = np.clip(controls, *task.ctrl_range.T)
        return controls, events

    def target_pose(self, event: dict) -> tuple:
        m, task, packet = self.model, self.task, self.packet
        g, k = event['geom2'], min(task.profile.horizon, round(event['start_time'] / packet['dt']))
        R = task.state.geom_xmat[g].reshape(3, 3).copy()
        pos = task.state.geom_xpos[g].copy()
        if task.name == 'sharpa_pen':
            a = packet['ref'][k, task.adr['axis']:task.adr['axis'] + 2]
            angle = np.arctan2(a[1], a[0])
            # Pen geom capsule axis local z lies along the pen's body x axis.
            R = np.array([[0., -np.sin(angle), np.cos(angle)], [0., np.cos(angle), np.sin(angle)], [-1., 0., 0.]])
            pos = packet['ref'][k, task.adr['pos']:task.adr['pos'] + 3]
        elif task.name == 'sharpa_coin':
            normal = np.asarray(packet['ref'][k, task.adr['normal']:task.adr['normal'] + 3], dtype=float).copy()
            normal /= max(np.linalg.norm(normal), 1e-12)
            x = np.array([1., 0., 0.]); x -= normal * (normal @ x); x /= max(np.linalg.norm(x), 1e-12)
            R = np.stack([x, np.cross(normal, x), normal], axis=1)
            pos = packet['ref'][k, task.adr['pos']:task.adr['pos'] + 3]
        return pos + R @ event['object_local_point'], R @ event['object_local_normal'], R

    def lm(self, geom: int, local: np.ndarray, normal: np.ndarray, target: np.ndarray,
           desired_normal: np.ndarray, reference: np.ndarray, base_qpos: np.ndarray | None = None) -> np.ndarray:
        return self.ik.solve(geom, local, normal, target, desired_normal, reference,
                             base_qpos if base_qpos is not None else self.state['qpos'])
