"""Native per-world contact loads, independent of the MPPI cost formula."""

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")
import warp as wp

from retarget.env.grasp.capture import capture_guard
from retarget.env.grasp.forceaware_mesh_audit import mesh_contact_snapshot
from retarget.env.grasp.spider_contact import ContactLoads, required_finger_mask
from retarget.env.grasp.spider_physics import BatchedPhysics


def contact_model():
    # A real free object and five servo fingers with fixed collision pads.
    # The additional palm-object contact must never appear in finger loads.
    fingers = "".join(f'''<body name="right_finger{i}_link1" pos="{(i-3)*.09} 0 .043">
      <joint name="finger{i}" type="slide" axis="0 0 1"/>
      <body name="pad{i}"><geom type="sphere" size=".025" mass=".05"/></body>
    </body>''' for i in range(1, 6))
    actuators = "".join(f'<position joint="finger{i}" kp="100" kv="2"/>' for i in range(1, 6))
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002" gravity="0 0 0" integrator="implicitfast"
          cone="pyramidal" jacobian="dense" iterations="20" ls_iterations="50" tolerance="1e-10"/>
      <worldbody>
        <body name="grasp_object"><freejoint name="obj_joint"/>
          <geom type="box" size=".25 .1 .02" mass="10"/></body>
        <body name="palm" pos="0 .08 .04"><geom type="sphere" size=".025"/></body>
        {fingers}
      </worldbody><actuator>{actuators}</actuator>
    </mujoco>''')


def test_ca_finger_requirements_deduplicate_and_include_fixed_pad_ancestry():
    cpu = contact_model()
    ids = [cpu.body(name).id for name in ("pad1", "right_finger3_link1", "pad1", "palm")]
    np.testing.assert_array_equal(required_finger_mask(cpu, ids), [True, False, True, False, False])
    np.testing.assert_array_equal(required_finger_mask(cpu, [0, cpu.body("palm").id]), np.zeros(5, bool))


@pytest.mark.parametrize("bodies", ([-1], [10000], [1.5], [[1, 2]]))
def test_ca_finger_requirements_reject_invalid_body_ids(bodies):
    with pytest.raises(ValueError, match="valid integer body IDs"):
        required_finger_mask(contact_model(), bodies)


@pytest.fixture(scope="module")
def native_contacts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for real native contact force sampling")
    wp.init(); wp.set_device("cuda:0")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        cpu = contact_model()
        physics = BatchedPhysics(cpu, 2, "cuda:0", nconmax=32, njmax=256, capture=False)
        ids = np.array([cpu.body(name).id for name in ("pad1", "pad1", "pad3", "palm")])
        loads = ContactLoads(physics, ids)
        physics.reset(cpu.qpos0, np.zeros(cpu.nv))
        physics.warmup()
        loads.sample()  # Compile all measurement kernels before graph capture.
        yield physics, loads, stream


def test_native_world_loads_match_independent_cpu_contact_force(native_contacts, record_property):
    physics, loads, _ = native_contacts
    cpu = physics.cpu
    controls = torch.tensor([[-.001]*5, [.001]*5], device=physics.device)
    physics.reset(cpu.qpos0, np.zeros(cpu.nv))
    physics.step(controls, use_graph=False)
    observed = loads.sample().clone()
    expected = []
    for command in controls.cpu().numpy():
        data = mujoco.MjData(cpu)
        data.ctrl[:] = command
        mujoco.mj_step(cpu, data)
        expected.append(mesh_contact_snapshot(cpu, data, object_body=cpu.body("grasp_object").id)
                        ["finger_normal_force_n"])
    expected = torch.as_tensor(np.asarray(expected), dtype=torch.float32, device=physics.device)
    torch.testing.assert_close(observed, expected, rtol=2e-3, atol=2e-4)
    record_property("cpu_gpu_normal_force_max_abs_error_n", float((observed-expected).abs().max()))
    assert bool((observed[0] > 0.).all())
    assert torch.count_nonzero(observed[1]) == 0
    assert not torch.equal(observed[0], observed[1])
    assert observed.shape == (2, 5) and observed.dtype == torch.float32
    assert loads.required_fingers.dtype == torch.bool
    torch.testing.assert_close(loads.required_fingers,
        torch.tensor([True, False, True, False, False], device=physics.device))


def test_no_object_contacts_clear_previous_loads_and_preserve_buffer(native_contacts):
    physics, loads, _ = native_contacts
    cpu = physics.cpu
    physics.reset(cpu.qpos0, np.zeros(cpu.nv))
    physics.step(torch.zeros((2, cpu.nu), device=physics.device), use_graph=False)
    buffer = loads.sample()
    assert bool((buffer > 0.).any())
    qpos = np.array(cpu.qpos0)
    qpos[0] += 2.  # Object free joint occurs before the five hand joints.
    physics.reset(qpos, np.zeros(cpu.nv))
    physics.step(torch.zeros((2, cpu.nu), device=physics.device), use_graph=False)
    cleared = loads.sample()
    assert cleared.data_ptr() == buffer.data_ptr()
    assert torch.count_nonzero(cleared) == 0


def test_captured_sampling_reads_live_controls_and_contact_count(native_contacts):
    physics, loads, _ = native_contacts
    cpu = physics.cpu
    qpos = torch.as_tensor(cpu.qpos0, dtype=torch.float32, device=physics.device).clone()
    qvel = torch.zeros(cpu.nv, device=physics.device)
    controls = torch.zeros((2, cpu.nu), device=physics.device)
    trace = torch.empty((2, 3, 5), device=physics.device)

    def run():
        physics.reset(qpos, qvel)
        for step in range(3):
            physics.step(controls, use_graph=False)
            trace[:, step].copy_(loads.sample())

    run()
    expected = trace.clone()
    with capture_guard(), wp.ScopedCapture() as capture:
        run()
    wp.capture_launch(capture.graph)
    torch.testing.assert_close(trace, expected, rtol=2e-4, atol=2e-4)
    controls.fill_(-.01)
    wp.capture_launch(capture.graph)
    assert not torch.allclose(trace, expected, rtol=1e-3, atol=1e-4)
    qpos[0] += 2.
    wp.capture_launch(capture.graph)
    assert torch.count_nonzero(trace) == 0
    qpos[0] -= 2.; controls.zero_()
    wp.capture_launch(capture.graph)
    torch.testing.assert_close(trace, expected, rtol=2e-4, atol=2e-4)
