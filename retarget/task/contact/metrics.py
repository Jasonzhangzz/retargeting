"""Acceptance copied from the source examples; never used as optimizer costs."""
import numpy as np
import mujoco


def assess(task, trajectory: dict, rows: list, complete: bool) -> dict:
    q, s = trajectory['qpos'], trajectory['sensordata']
    finite = bool(np.isfinite(q).all() and np.isfinite(trajectory['qvel']).all())
    result = dict(complete=complete, finite=finite, task_success=False, task=task.name,
                  evaluation='reference_tracking' if task.reference is not None else 'task_generation')
    if not len(q):
        return result
    if task.name == 'sharpa_pen':
        a = task.adr['axis']
        angle = np.unwrap(np.arctan2(s[:, a + 1], s[:, a]))
        turns = (angle[-1] - angle[0]) / (2 * np.pi)
        dropped = bool(np.any(q[:, 24] < task.initial['qpos'][24] - .03))
        result.update(turns=float(turns), dropped=dropped, task_success=bool(complete and finite and turns >= 3 and not dropped))
    elif task.name == 'h1_jog':
        dt = task.model.opt.timestep
        v = s[:, task.adr['torso_vel']:task.adr['torso_vel'] + 2]
        target = np.c_[np.minimum(2 * np.arange(len(q)) * dt / 3, 2), np.zeros(len(q))]
        command_error = float(np.mean(np.sum((v - target)**2, axis=1) + (s[:, task.adr['torso_gyro'] + 2] * np.pi / 180)**2))
        error = command_error
        if task.reference is not None:
            reference_sensor = task.reference['sensordata'][1:len(q) + 1]
            target = reference_sensor[:, task.adr['torso_vel']:task.adr['torso_vel'] + 2]
            gyro = s[:, task.adr['torso_gyro'] + 2] - reference_sensor[:, task.adr['torso_gyro'] + 2]
            error = float(np.mean(np.sum((v - target)**2, axis=1) + (gyro * np.pi / 180)**2))
        fell = bool(s[:, task.adr['pelvis_zaxis'] + 2].min() < .3 or s[:, task.adr['torso_pos'] + 2].min() < .5)
        result.update(velocity_tracking_error=error, fell=fell, final_speed=float(v[-50:, 0].mean()),
                      velocity_tracking_target='frozen_robot_trajectory' if task.reference is not None else '2m_s_command_ramp',
                      source_command_velocity_error=command_error,
                      travelled_m=float(q[-1, 0] - task.initial['qpos'][0]),
                      task_success=bool(complete and finite and error < .025 and not fell))
        if task.reference is not None:
            ref = task.reference['qpos'][1:len(q) + 1]
            result['root_position_rms_m'] = float(np.sqrt(np.mean(np.sum((q[:, :3] - ref[:, :3])**2, axis=1))))
            from retarget.task.humanoid.metrics import assess_trajectory
            tracking = assess_trajectory(task.model, q, ref, complete, humanoid_object=False)
            result['tracking'] = tracking
            ref_sensor = task.reference['sensordata'][1:]
            command = np.c_[np.minimum(2 * np.arange(1, len(ref_sensor) + 1) * dt / 3, 2), np.zeros(len(ref_sensor))]
            ref_velocity = ref_sensor[:, task.adr['torso_vel']:task.adr['torso_vel'] + 2]
            result['reference_source_command_velocity_error'] = float(np.mean(np.sum((ref_velocity - command)**2, axis=1) + (ref_sensor[:, task.adr['torso_gyro'] + 2] * np.pi / 180)**2))
            result['joint_position_rms_rad'] = float(np.sqrt(np.mean((q[:, 7:] - ref[:, 7:])**2)))
            result['joint_velocity_rms_rad_s'] = float(np.sqrt(np.mean((trajectory['qvel'][:, 6:] - task.reference['qvel'][1:len(q) + 1, 6:])**2)))
            result['task_success'] &= tracking['task_success']
    else:
        normal = s[:, task.adr['normal']:task.adr['normal'] + 3]
        touch = np.zeros((len(q), 6), bool)
        overlap, penetration = 0., 0.
        free, drops = 0, []
        obj = task.model.geom('coin').id
        # Original coin acceptance checks hand self-collision in a separate
        # model, with the coin masked out, independently of execution contacts.
        hand_only = task.scene.copy()
        hand_only.geom('coin').contype = hand_only.geom('coin').conaffinity = 0
        check = hand_only.compile()
        cd = mujoco.MjData(check)
        for t, row in enumerate(rows):
            for c in row['contacts']:
                g1, g2 = c['geom1'], c['geom2']
                if obj in (g1, g2):
                    touch[t, task.owner[g1 + g2 - obj]] |= c['force'][0] > .002
                    penetration = max(penetration, -c['geometric_distance'])
                else:
                    overlap = max(overlap, -c['geometric_distance'])
            cd.qpos[:] = q[t]
            mujoco.mj_kinematics(check, cd)
            mujoco.mj_collision(check, cd)
            overlap = max(overlap, -cd.contact.dist[:cd.ncon].min(initial=0.))
            free = 0 if touch[t].any() else free + 1
            if free * task.model.opt.timestep >= .3 or q[t, 24] < q[0, 24] - .05:
                drops.append(t)
        k, reach = 0, [0]
        order = [0, 1, 2, 3, 2, 1]
        for t in range(len(q)):
            f = order[(k + 1) % 6]
            if touch[t, f + 1] and abs(q[t, 23] - task.lines[f]) <= .006 and (-1)**(k + 1) * normal[t, 2] >= np.sqrt(.5):
                k += 1; reach.append(t)
        flat = []
        for j, (t0, t1) in enumerate(zip(reach, reach[1:] + [len(q)])):
            if t1 <= t0:
                continue
            f = order[j % 6]
            ok = touch[t0:t1, f + 1] & (np.abs(q[t0:t1, 23] - task.lines[f]) <= .006)
            flat.append(normal[t0 + np.where(ok, (-1)**j * normal[t0:t1, 2], -2.).argmax()])
        turns = [float(np.degrees(np.arccos(np.clip(u @ v, -1, 1)))) for u, v in zip(flat, flat[1:])]
        jitter = float(np.sqrt(np.mean(np.diff(q[::4, :22], 2, axis=0)**2)) / .02**2) if len(q[::4]) >= 3 else None
        trip = k >= 6 and len(turns) >= 6 and min(turns[:6]) >= 150 and penetration <= .002 and (not drops or drops[0] > reach[6])
        result.update(goals=k, flip_angles_deg=turns, penetration_m=penetration, self_overlap_m=overlap,
            first_drop_frame=drops[0] if drops else None, jitter_rad_s2=jitter,
            task_success=bool(complete and finite and trip and jitter is not None and jitter <= 5 and overlap <= .001))
    contacts = [c for row in rows for c in row['contacts']]
    result.update(peak_contact_force_n=max((float(c['force'][0]) for c in contacts), default=0.),
        peak_friction_violation_n=max((c['friction_violation'] for c in contacts), default=0.),
        mean_slip_m_s=float(np.mean([c['slip'] for c in contacts])) if contacts else 0.)
    return result
