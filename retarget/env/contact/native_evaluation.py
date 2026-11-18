"""One stateless native speculative rollout; safe to dispatch to isolated CPU workers."""
import mujoco
import numpy as np
import time
from .native import restore, step_cost, forecast_contacts
from .prediction import predict_pair


def evaluate_candidate(model: mujoco.MjModel, state: dict, controls: np.ndarray,
                       packet: dict, events: list, dynamics_cost: bool) -> dict:
    scratch = mujoco.MjData(model)
    started = time.perf_counter()
    forecast = forecast_contacts(model, state, controls, packet, events) if dynamics_cost else []
    forecast_seconds = time.perf_counter() - started
    started = time.perf_counter()
    restore(model, scratch, state)
    initial_data = mujoco.MjData(model)
    mujoco.mj_copyData(initial_data, model, scratch)
    geom_positions = [scratch.geom_xpos.copy()]
    geom_quats = [scratch.geom_xmat.copy()]
    qs, vs = ([scratch.qpos.copy()], [scratch.qvel.copy()])
    total, sums, first_hits, predictions = (0.0, {}, {}, [])
    previous = scratch.ctrl.copy()
    for t, u in enumerate(controls):
        scratch.ctrl[:] = u
        pre_q, pre_v = scratch.qpos.copy(), scratch.qvel.copy()
        mujoco.mj_step(model, scratch)
        post_q, post_v = scratch.qpos.copy(), scratch.qvel.copy()
        # mj_step's derived fields and force belong to the integration's input
        # state. Score the additional terms at that same stage, as source sensor
        # costs already do. Never re-forward and replace the force actually used.
        scratch.qpos[:], scratch.qvel[:] = pre_q, pre_v
        cost, parts, records = step_cost(model, scratch, packet, t, events, dynamics_cost, terminal=t == len(controls) - 1)
        scratch.qpos[:], scratch.qvel[:] = post_q, post_v
        if not (np.isfinite(post_q).all() and np.isfinite(post_v).all()):
            cost = np.inf
        for i, e in enumerate(events):
            touching = any(({r['geom1'], r['geom2']} == {e['geom1'], e['geom2']} and r['force'][0] > 0.002 for r in records))
            if touching and i not in first_hits:
                first_hits[i] = t * model.opt.timestep
        smooth = 0.001 * float(np.sum((u - previous) ** 2)) if dynamics_cost else 0.0
        previous = u.copy()
        total += cost + smooth
        parts['smooth'] += smooth
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        if not np.isfinite(total):
            break
        mujoco.mj_kinematics(model, scratch)
        geom_positions.append(scratch.geom_xpos.copy())
        geom_quats.append(scratch.geom_xmat.copy())
        qs.append(scratch.qpos.copy())
        vs.append(scratch.qvel.copy())
    if np.isfinite(total):
        if dynamics_cost:
            for i, e in enumerate(events):
                delta = first_hits.get(i, len(controls) * model.opt.timestep) - e['start_time']
                penalty = 0.1 * delta ** 2
                total += penalty
                sums['timing'] += penalty
    dynamics_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if np.isfinite(total) and events:
        from scipy.spatial.transform import Rotation
        from retarget.env.contact.prediction import ray_occluded
        for e in events:
            gids = [e['geom1'], e['geom2']]
            pos = np.asarray(geom_positions)[:, gids]
            mats = np.asarray(geom_quats)[:, gids].reshape(-1, 3, 3)
            quats = Rotation.from_matrix(mats).as_quat()[:, [3, 0, 1, 2]].reshape(-1, 2, 4)
            prediction = predict_pair(model, initial_data, *gids, np.arange(len(pos)) * model.opt.timestep, pos, quats, clearance=float(model.geom_margin[gids].sum()))
            predictions.append(prediction.as_dict())
    for item in forecast:
        if item['status'] != 'uncertain':
            continue
        # A commanded-pose approximation is not a physical infeasibility
        # certificate. Preserve UNKNOWN, and let the mandatory native rollout
        # (which rebuilds speculative constraints every step) adjudicate cost.
        pair = {item['geom1'], item['geom2']}
        hits = [first_hits[i] for i, event in enumerate(events)
                if {event['geom1'], event['geom2']} == pair and i in first_hits]
        realized = next((p['status'] for p in predictions if {p['geom1'], p['geom2']} == pair), None)
        item['evaluation_fallback'] = 'native_speculative_rollout'
        item['native_first_contact_time'] = min(hits) if hits else None
        item['realized_geometry_status'] = realized
        item['fallback_result'] = ('invalid_rollout' if not np.isfinite(total) else
                                   'contact_constraint' if hits else
                                   'geometry_' + realized if realized in ('hit', 'no_collision') else 'engine_only')
    return dict(cost=total, parts=sums, first_hits=first_hits, predictions=forecast,
                realized_predictions=predictions, qpos=np.asarray(qs), qvel=np.asarray(vs),
                phase_seconds=dict(command_forecast=forecast_seconds, dynamics_and_cost=dynamics_seconds,
                                   realized_prediction=time.perf_counter() - started))
