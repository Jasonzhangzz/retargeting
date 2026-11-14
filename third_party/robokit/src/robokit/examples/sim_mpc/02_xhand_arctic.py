# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: two XHands hold an articulated ARCTIC notebook up and turn its cover, tracking a clip robokit retargets.

ARCTIC and MANO are licensed, so bring your own:
- ARCTIC (https://arctic.is.tue.mpg.de): `--arctic-root` holds `raw_seqs/s05` and `meta/object_vtemplates/notebook`.
- MANO (https://mano.is.tue.mpg.de): `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` in `~/.robokit/body_models/mano`.

SPIDER: https://arxiv.org/abs/2511.09484.
"""

import time
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
import warp as wp

from robokit.assets import fetch
from robokit.assets.robots.end_effectors import xhand
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandSpec
from robokit.helpers.hand_retargeting.presets._common import FIVE_FINGERTIPS
from robokit.helpers.hand_retargeting.presets.xhand import _LINKS, offline, spec
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.robo import Robot
from robokit.sim import MjWarpSim
from robokit.smplx import BodyModelSpecTensors, BodyModelState, body_lbs_warp, load_mano
from robokit.terms import StateTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from robokit.xform.numpy.rotation_conversions import matrix_to_euler_angles


REPO, REVISION = "retarget/retarget_example", "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"  # SPIDER's released data
SCENE, HANDS = "processed/arcticv2/xhand/bimanual/s01-box_use_01", "processed/gigahand/assets"  # XHands: in GigaHands
SEQUENCE, FRAMES = "s05/notebook_use_01", slice(305, 380)  # 2.5 s at 30 Hz
FLOOR_Z = 1.0924  # m in ARCTIC's world: the lowest point of the clip's hands
LID_FRICTION = 0.17  # N m, assumed: near gravity's peak, so the cover barely falls
R_MANO_PALM = dict(  # SPIDER's offsets from MANO's wrist to the XHand's palm
    right=np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    left=np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
)
TARGETS = ("wrist", *FIVE_FINGERTIPS)


def main(arctic_root: Path):
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: the object's pose and lid angle, MANO's parameters for each hand, SPIDER's XHand scene ---
    root = fetch([f"{SCENE}/scene.xml", f"{HANDS}/robots/xhand/**"], REVISION, REPO)
    arctic = np.load(arctic_root / f"raw_seqs/{SEQUENCE}.object.npy", allow_pickle=True).astype(float)
    mano = np.load(arctic_root / f"raw_seqs/{SEQUENCE}.mano.npy", allow_pickle=True).item()
    lid_range, arctic = (arctic[:, 0].min(), arctic[:, 0].max()), arctic[FRAMES]  # angle, axis-angle, translation in mm
    num_frames = len(arctic)
    print(f"loaded {num_frames} frames of {SEQUENCE}")
    # each quaternion in the previous one's hemisphere, so interpolating them is safe
    continuous = lambda q: q * np.cumprod(np.r_[1.0, np.sign(np.sum(q[1:] * q[:-1], 1))])[:, None]

    # --- scene: both XHands on six position-controlled base joints each, SPIDER's contact pairs and solver settings; its box and unused massless body give way to a free base with the lid hinged to it, one convex hull each ---
    scene = mujoco.MjSpec.from_file(str(root / SCENE / "scene.xml"))
    pairs = [(p.geomname1, p.condim, p.friction.copy()) for p in scene.pairs if p.geomname2 == "right_object_0"]
    for element in [*(p for p in scene.pairs if "_object_" in p.geomname2), scene.body("right_object"),
                    scene.body("left_object"), *(m for m in scene.meshes if m.file.startswith("objects"))]:  # fmt: skip
        scene.delete(element)
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = str(root / HANDS / mesh.file)
    scene.meshdir = ""
    base = scene.worldbody.add_body(name="base")
    base.add_joint(name="object_base", type=mujoco.mjtJoint.mjJNT_FREE, armature=1e-4, frictionloss=1e-4)
    lid = base.add_body(name="lid")
    lid.add_joint(name="object_lid", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, -1], range=lid_range,
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, armature=1e-4, frictionloss=LID_FRICTION)  # fmt: skip
    for part, file, body in [("base", "bottom", base), ("lid", "top", lid)]:
        scene.add_mesh(name=part, file=str(arctic_root / f"meta/object_vtemplates/notebook/{file}.obj"),
                       scale=[1e-3] * 3, maxhullvert=32, inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)  # fmt: skip
        body.add_geom(name=part, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part, density=1000, rgba=[1, 1, 1, 1])
        for geom, condim, friction in pairs:
            scene.add_pair(geomname1=geom, geomname2=part, condim=condim, friction=friction)
    # softer fingers (kp / 30) and wrists (kp / 10), inertia to match: the same motion, far gentler on contact
    wrist = lambda name: "forearm_" in name and "forearm_t" not in name
    for joint in scene.joints:
        joint.armature = 0.033 if "_hand_" in joint.name else 0.1 if wrist(joint.name) else joint.armature
    for actuator in scene.actuators:
        if "forearm_t" not in actuator.name:
            kp = 100 if wrist(actuator.name) else 10
            actuator.gainprm[0], actuator.biasprm[1] = kp, -kp
    o = scene.option
    o.timestep, o.iterations, o.integrator = 0.01, 20, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = scene.compile()
    print(f"scene: {model.nq} qpos, {model.nu} actuators, {model.npair} contact pairs")

    # --- retarget: MANO's wrist and fingertips to each XHand's base pose and 12 joints ---
    ref = np.zeros((num_frames, model.nq))
    for side in ("right", "left"):
        hand, mano_spec = mano[side], load_mano(side=side)  # ARCTIC poses are relative to MANO's mean hand
        rot, pose, trans = (np.asarray(hand[k], float)[FRAMES] for k in ("rot", "pose", "trans"))
        full_pose = np.c_[rot, pose + mano_spec.metadata["hands_mean"]].reshape(num_frames, 16, 3)
        mano_state = BodyModelState(np.tile(hand["shape"], (num_frames, 1)), full_pose, trans)
        posed = body_lbs_warp(BodyModelSpecTensors(mano_spec, str(wp.get_preferred_device())), mano_state)
        lift = [0.0, 0.0, -FLOOR_Z]
        wrist, tips = posed["T_world_joint"].numpy()[:, 0, :3] + lift, posed["landmarks"].numpy() + lift
        q_wxyz = continuous(matrix_to_quaternion(axis_angle_to_matrix(rot) @ R_MANO_PALM[side]))
        robot = Robot.load(str(getattr(xhand, f"URDF_PATH_{side.upper()}")))
        links = {n: _LINKS[n].replace("right_", f"{side}_") for n in TARGETS}
        hand_spec = HandSpec(
            floating_base=True, target_names=TARGETS, target_link_names=links,
            target_chains=tuple(("wrist", tip) for tip in FIVE_FINGERTIPS), root_target_name="wrist",
            contact_target_names=FIVE_FINGERTIPS, root_link_coord_spec=spec.root_link_coord_spec,
            target_coord_spec=spec.target_coord_spec if side == "right" else HandCoordinateSpec("-y", "x"),
        )  # fmt: skip
        points = np.concatenate([wrist[:, None], tips], 1)
        packed = HandRetargeterOffline(offline, robot, hand_spec).solve_numpy(points, q_wxyz)
        T_world_link = robot.forward_kinematics_numpy(packed[:, 7:], packed[:, :7])
        T_world_hand = T_world_link[:, robot.link_names.index(links["wrist"])]  # the scene's hand body
        base_adr = model.jnt_qposadr[model.joint(f"{side[0].upper()}_forearm_tx_link_joint").id]
        ref[:, base_adr : base_adr + 3] = T_world_hand[:, :3]
        euler = np.unwrap(matrix_to_euler_angles(quaternion_to_matrix(T_world_hand[:, 3:]), "rzxy"), axis=0)
        ref[:, base_adr + 3 : base_adr + 6] = euler * [1, 1, -1]  # roll about z, pitch about x, yaw about -y
        ref[:, model.jnt_qposadr[[model.joint(n).id for n in robot.spec.actuated_joint_names]]] = packed[:, 7:]
    obj = model.nq - 8  # the base's free joint, then the lid's hinge
    ref[:, obj : obj + 3] = arctic[:, 4:7] / 1000 + [0.0, 0.0, -FLOOR_Z]
    ref[:, obj + 3 : obj + 7] = continuous(matrix_to_quaternion(axis_angle_to_matrix(arctic[:, 1:4])))
    ref[:, -1] = arctic[:, 0]

    # --- reference: upsampled to the 10 ms sim step, then held while the horizon runs past the clip ---
    end = len(ref) - 1
    t = np.r_[np.arange(0, end, 0.3), np.full(300, end)]
    qpos_ref = np.stack([np.interp(t, np.arange(len(ref)), c) for c in ref.T], 1)
    qpos_ref[:, obj + 3 : obj + 7] /= np.linalg.norm(qpos_ref[:, obj + 3 : obj + 7], axis=1, keepdims=True)
    ctrl_ref = qpos_ref[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    state = mujoco.MjData(model)
    state.qpos[:] = qpos_ref[0]
    for _ in range(100):  # XHands are thicker than human hands: contact pushes them out of the object, pinned here
        state.qpos[obj:], state.qvel[-7:], state.ctrl[:] = qpos_ref[0, obj:], 0, ctrl_ref[0]
        mujoco.mj_step(model, state)
    state.time, state.qvel[:] = 0.0, 0.0
    mujoco.mj_forward(model, state)

    # --- SimMPC: track the hands, the object and its lid, weighting the object most, and reward each hand for touching the object; a 1 s plan refined 6 times, replanned every 0.25 s ---
    names = [model.joint(model.dof_jntid[i]).name for i in range(model.nv)]
    weight = [1.0 if "object" in n else 0.1 if "forearm_t" in n else 0.03 if "forearm" in n else 0.003 for n in names]
    weight[-4:-1] = [0.3] * 3  # the base's rotation
    geoms = [model.geom(i).name for i in range(model.ngeom)]
    target, touch = np.array([g in ("base", "lid") for g in geoms]), np.ones(len(qpos_ref))
    hands = [np.array([g.startswith(f"collision_hand_{side}") for g in geoms]) for side in ("right", "left")]
    contacts = [(hand, target, touch, 0.05) for hand in hands]
    cost = StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), np.array(weight), contacts=contacts)
    ctrl_std = np.where(["forearm" in model.actuator(i).name for i in range(model.nu)], 0.02, 0.2)
    config = SimMPCConfig(
        num_samples=1024, horizon_steps=100, node_steps=25, ctrl_steps=25,
        stages=[SimMPCStage(5, [(0.25, None), (1.0, 0.02)]), SimMPCStage(1)],  # 1.33x faster
        min_improvement=-1.0,
        std_growth=2.0, noise_decay=0.3, terminal_scale=1.0, ctrl_std=ctrl_std,
    ).add(cost)  # fmt: skip
    steps = (len(qpos_ref) - 300) // config.ctrl_steps * config.ctrl_steps  # the clip, not the hold past it
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=256, njmax=768)
    mpc = SimMPC(config, sim, ctrl_ref)
    print(f"planning {steps / 100:.1f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: print(f"  t={plans[0]['t0'] / 100:.1f} s: cost {plans[0]['plan_cost']:.3f}")
    (rollout,), _ = mpc.execute(sim, steps, callback=report)

    # --- check: SPIDER's success, a mean object error within 10 cm and 0.5 rad; held up; the lid against the human's ---
    q, q_ref = rollout["qpos"][:, obj + 3 : obj + 7], qpos_ref[:steps, obj + 3 : obj + 7]
    position = np.linalg.norm(rollout["qpos"][:, obj : obj + 3] - qpos_ref[:steps, obj : obj + 3], axis=1)
    rotation = 2 * np.arccos(np.clip(np.abs(np.sum(q * q_ref, 1)), 0, 1))
    lid_error = np.abs(rollout["qpos"][:, -1] - qpos_ref[:steps, -1])
    dropped = bool(np.any(rollout["qpos"][:, obj + 2] < qpos_ref[:steps, obj + 2] - 0.1))  # 10 cm under the human's
    print(
        f"notebook: {'pass' if position.mean() <= 0.1 and rotation.mean() <= 0.5 and not dropped else 'fail'}, "
        f"{'dropped' if dropped else 'held up'}, mean/max error {100 * position.mean():.1f}/{100 * position.max():.1f} "
        f"cm, {rotation.mean():.2f}/{rotation.max():.2f} rad, lid {lid_error.mean():.2f}/{lid_error.max():.2f} rad"
    )
    print(f"planned in {time.perf_counter() - start:.1f} s")

    # --- viewer: replay the simulated motion, or the kinematic reference it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=1.0, height=1.0)
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(0, steps, 2):
            viewer.update_qpos(qpos_ref[k] if reference.value else rollout["qpos"][k])
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    tyro.cli(main)
