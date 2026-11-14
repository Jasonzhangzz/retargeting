"""Object-pose/no-drop evaluation independent of force magnitude thresholds."""
import mujoco
import numpy as np
import warp as wp


@wp.kernel
def mark_hand_contact(count: wp.array(dtype=int), worlds: wp.array(dtype=int),
                      geoms: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float),
                      object_geom: int, proximity: float, out: wp.array(dtype=int)):
    i = wp.tid()
    if i < count[0] and dist[i] <= proximity:
        pair = geoms[i]
        if pair[0] == object_geom or pair[1] == object_geom:
            wp.atomic_max(out, worlds[i], 1)


def drop_event(relative_z, proximity_contact, *, hz=300, slip_m=.05, gap_seconds=.1):
    """Operational drop: downward object/hand separation and sustained contact loss."""
    relative_z = np.asarray(relative_z)
    present = np.asarray(proximity_contact, bool)
    if relative_z.shape != present.shape or len(relative_z) < 2:
        raise ValueError('Expected initial and post-state contact/relative-height arrays')
    slipped = relative_z[0] - relative_z[1:] > slip_m
    condition = slipped & ~present[1:]
    run, longest, first = 0, 0, None
    minimum = int(np.ceil(hz * gap_seconds))
    for i, value in enumerate(condition):
        run = run + 1 if value else 0
        longest = max(longest, run)
        if first is None and run >= minimum:
            first = (i + 1) / hz
    return dict(dropped=first is not None, first_detection_seconds=first,
                longest_slipped_no_contact_seconds=longest/hz,
                maximum_downward_slip_mm=float(max(0., np.max(relative_z[0]-relative_z[1:]))*1000),
                slip_threshold_m=slip_m, sustained_loss_seconds=gap_seconds)


def evaluate_drop(scene, initial_qpos, trajectory):
    """Only MJWarp GPU kinematics/collision; query margins do not affect execution."""
    from robokit.sim import MjWarpSim
    import mujoco_warp as mjw
    model = mujoco.MjModel.from_xml_path(str(scene))
    # Query a 2 mm proximity band. Execution model and saved states are untouched.
    # MJWarp multi-point CCD requires zero margins. A single-point distance
    # query suffices for presence; disable MULTICCD and native CCD in this copy.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    model.geom_margin[:] = .002
    q = np.concatenate([np.asarray(initial_qpos)[None], np.asarray(trajectory)]).astype(np.float32)
    n = len(q)
    with wp.ScopedDevice('cuda:0'):
        sim = MjWarpSim(model, n, mujoco.MjData(model), nconmax=128, njmax=2, device='cuda:0')
        with wp.ScopedStream(sim.stream):
            sim.data.qpos.assign(q)
            sim.kinematics()
            mjw.collision(sim._m, sim._d)
            d = sim._d
            contact = wp.zeros(n, dtype=int, device='cuda:0')
            wp.launch(mark_hand_contact, dim=d.contact.dist.shape[0], inputs=[d.nacon, d.contact.worldid,
                      d.contact.geom, d.contact.dist, model.geom('tracked_object_geom').id, .002, contact])
            wp.synchronize_stream(sim.stream)
            hand_z = d.xpos.numpy()[:, model.body('right_palm_link').id, 2]
            present = contact.numpy().astype(bool)
    report = drop_event(q[:, 28] - hand_z, present)
    report.update(contact_proximity_m=.002, poststate_contact_fraction=float(present[1:].mean()),
                  backend='MJWarp GPU collision including palm',
                  query_mode='Single-point convex distance; MULTICCD and native CCD disabled only in query copy',
                  definition='Object relative to palm slips down >50 mm with no hand surface within 2 mm for >=100 ms')
    return report
