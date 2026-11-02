"""Real-mesh regressions for centered distal anchors and both optimizers."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("rtree")

from retarget.env.grasp.contact_regions import (
    adapt_contact_regions,
    contact_region_diagnostics,
    region_coordinates,
    region_mask,
    region_model,
    region_pool,
    region_preference,
    surface_pool,
    template_regions,
)
from retarget.env.grasp.graspadmm_scene import project_to_hand
from retarget.env.grasp.hand import HAND_BODIES, query_path
from retarget.task.grasp.dexonomy import load_template, map_contacts


@pytest.mark.parametrize("template,count", [("6_Prismatic_4_Finger", 5),
    ("7_Prismatic_3_Finger", 4), ("8_Prismatic_2_Finger", 3)])
def test_template_anchors_move_from_tips_to_palmar_centers(template, count):
    model = region_model()
    root = Path(__file__).resolve().parents[1]/"data/grasp/templates"
    shadow, _, contacts, _ = load_template(root, template)
    bodies, local, normals, mapping = map_contacts(shadow, contacts, model)
    original = local.copy()
    points, facing, records, region = adapt_contact_regions(model, template, bodies, local, normals, mapping)
    assert len(points) == len(region["bodies"]) == count
    np.testing.assert_array_equal(local, original)
    for i, name in enumerate(region["bodies"]):
        assert region_coordinates(name, original[i], normals[i])[0] > .85
        assert region_mask(name, points[i], facing[i], region)
        assert abs(region_coordinates(name, points[i], facing[i])[0] - .5) < .08
        assert records[i]["original_wuji_local"] == mapping[i]["wuji_local"]
        with np.load(query_path(model, name)) as raw:
            np.testing.assert_array_equal(points[i], raw["local_pos"][records[i]["query_index"]])


def test_non_target_templates_keep_the_original_mapping_objects():
    points = np.zeros((1,3));normals = np.ones((1,3));mapping = [{}]
    model = region_model()
    result = adapt_contact_regions(model,"4_Adducted_Thumb",[model.body("right_finger2_link4").id],points,normals,mapping)
    assert result[0] is points and result[1] is normals and result[2] is mapping
    assert result[3] is None


def test_region_empty_raises_without_full_link_fallback():
    region = template_regions("6_Test", ["right_finger2_link4"])
    with pytest.raises(ValueError, match="Empty"):
        region_pool("right_finger2_link4", dict(region, u_min=4, u_max=5))


def test_admm_soft_projection_encourages_center_but_allows_tip_when_geometry_wins():
    model = region_model()
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[[.12, -.03, .1, .4, -.2, .3], np.full(20, .1)]
    mujoco.mj_kinematics(model, data)
    name = "right_finger2_link4"
    region = template_regions("6_Test", [name])
    env = SimpleNamespace(_model=model, _data=data, _cfg=SimpleNamespace(hand_prefix=""), contact_regions=region)
    with np.load(query_path(model, name)) as raw:
        p,n = raw["local_pos"],raw["local_normal"]
        u,_,_ = region_coordinates(name,p,n)
        tip = p[np.argmax(u)]
    body = model.body(name).id
    rotation = data.xmat[body].reshape(3,3)
    target = tip@rotation.T + data.xpos[body]
    world,local = project_to_hand(target[None], [name], env, [], [], "collision")
    pool,normals,_ = surface_pool(name)
    selected = np.argmin(np.linalg.norm(pool-local[0],axis=1))
    assert region_coordinates(name,local[0],normals[selected])[0] > .7
    assert np.isfinite(region_preference(name,local[0],normals[selected],region))
    np.testing.assert_allclose(world[0],local[0]@rotation.T+data.xpos[body])
    report = contact_region_diagnostics(data.qpos,[name],np.c_[world,normals[selected][None]@rotation.T],region)
    assert report["mode"] == "soft" and not report["semantic_success"]
    center_points,center_normals,_=region_pool(name,region)
    center=int(np.argmin(region_preference(name,center_points,center_normals,region)))
    target=center_points[center]@rotation.T+data.xpos[body]
    _,local=project_to_hand(target[None],[name],env,[],[],"collision")
    assert abs(region_coordinates(name,local[0],center_normals[center])[0]-.5) < .1


def cpu_objective():
    pytest.importorskip("mujoco_warp")
    from retarget.env.grasp.graspadmm_mppi import LocalPoseObjective, RefinementSettings
    objective = LocalPoseObjective.__new__(LocalPoseObjective)
    objective.cpu = region_model()
    objective.device = "cpu"
    objective.settings = RefinementSettings(points_per_body=16)
    objective.default_pools = {}
    for name in HAND_BODIES:
        with np.load(query_path(objective.cpu,name)) as raw:
            objective.default_pools[name] = (raw["local_pos"][::16],raw["local_normal"][::16])
    objective.low = torch.full((26,),-10.)
    objective.high = torch.full((26,),10.)
    return objective


def test_mppi_keeps_whole_distal_candidates_and_uses_finite_center_preference():
    from retarget.env.grasp.graspadmm_mppi import local_to_native
    o = cpu_objective();name = "right_finger2_link4"
    region = template_regions("6_Test",[name])
    q = np.zeros(26);q[6:] = .1
    o.bind(local_to_native(q), [[name]], contact_regions=region)
    sl = o.slices[name]
    points = o.local[sl].numpy();normals = o.normals[sl].numpy()
    assert not region_mask(name,points,normals,region).all()
    np.testing.assert_array_equal(o.groups[0].numpy(),np.arange(sl.start,sl.stop))
    # Required contact and wrench use the same finite biased score. A much
    # better geometric match can still choose a fingertip sample.
    u=region_coordinates(name,points,normals)[0]
    assert max(u) > .9
    penalty=o.region_penalty[sl].numpy()
    assert np.isfinite(penalty).all()
    assert abs(u[penalty.argmin()]-.5) < .1
    tip=np.flatnonzero(u>.9)[0]
    geometry=np.full(len(penalty),100.);geometry[tip]=0.
    assert (geometry+penalty).argmin()==tip
    o.bind(local_to_native(q),[[name]])
    np.testing.assert_array_equal(o.local[o.slices[name]].numpy(),o.default_pools[name][0])
    assert max(region_coordinates(name,o.local[o.slices[name]].numpy(),o.normals[o.slices[name]].numpy())[0]) > .9
    assert torch.equal(o.region_penalty,torch.zeros_like(o.region_penalty))


def test_mppi_projection_freezes_rotation_but_allows_translation_and_fingers():
    from retarget.env.grasp.graspadmm_mppi import local_to_native
    o = cpu_objective();q = np.r_[[.01,.02,.03,.2,-.3,.4],np.full(20,.1)]
    grasp = local_to_native(q)
    o.bind(grasp,[["right_finger2_link4"]],reference_orientation_wxyz=grasp[3:7].tolist())
    candidate = o.reference.clone();candidate += .005
    constrained = o.project(candidate)
    np.testing.assert_array_equal(constrained[3:6].numpy(),o.reference[3:6].numpy())
    assert not torch.equal(constrained[:3],o.reference[:3])
    assert not torch.equal(constrained[6:],o.reference[6:])
    wrong = grasp.copy();wrong[3:7] = [1,0,0,0]
    with pytest.raises(ValueError,match="locked reference"):
        o.bind(wrong,[["right_finger2_link4"]],reference_orientation_wxyz=grasp[3:7].tolist())
