"""Semantic template patches use intrinsic frames and real surface samples."""

import copy
import json

import mujoco
import numpy as np
import pytest

from retarget.config.run import ROOT
from retarget.env.grasp import semantic_patches as patches
from retarget.env.grasp.hand import query_path
from retarget.task.grasp.dexonomy import load_template


ASSETS = ROOT / "data/grasp/assets/wuji2/right.xml"
TEMPLATES = ROOT / "data/grasp/templates"
pytestmark = pytest.mark.skipif(not ASSETS.exists(), reason="Grasp assets unavailable")


@pytest.fixture(scope="module")
def wuji():
    return mujoco.MjModel.from_xml_path(str(ASSETS))


def _match(wuji, name):
    shadow, _, contacts, _ = load_template(TEMPLATES, name)
    return contacts, patches.match_semantic_patches(shadow, contacts, wuji)


def test_large_diameter_correspondences_remain_on_fixed_real_patches(wuji):
    shadow, _, contacts, _ = load_template(TEMPLATES, "1_Large_Diameter")
    saved = copy.deepcopy(contacts)
    bodies, local, normals, records, masks = patches.match_semantic_patches(shadow, contacts, wuji)
    assert len(contacts) == len(bodies) == len(masks) == 16
    for body, point, normal, record, patch, original, preserved in zip(
        bodies, local, normals, records, masks, contacts, saved
    ):
        np.testing.assert_array_equal(original[1], preserved[1])
        with np.load(query_path(wuji, int(body))) as pool:
            index = record["query_index"]
            assert patch["mask"].dtype == bool
            assert patch["mask"].shape == (len(pool["local_pos"]),)
            assert patch["mask"].any() and patch["mask"][index]
            assert patch["mask"].sum() < len(pool["local_pos"])
            np.testing.assert_allclose(point, pool["local_pos"][index], atol=1e-12)
            np.testing.assert_allclose(normal, pool["local_normal"][index], atol=1e-6)
            if record["wuji_body"] == "right_palm_link":
                assert pool["source_body"][index] == "right_palm_link"
        assert record["visibility_filtered"]
        assert len(record["source_chart_scales_m"]) == 3
        assert len(record["target_chart_scales_m"]) == 3
        json.dumps(record)


def test_thumb_normal_uses_bone_chart_and_proximal_uses_long_phalanx(wuji):
    _, (_, _, _, records, _) = _match(wuji, "1_Large_Diameter")
    by_source = {record["shadow_body"]: record for record in records}
    assert by_source["rh_ffproximal"]["wuji_body"] == "right_finger2_link2"
    assert wuji.joint(wuji.body_jntadr[wuji.body("right_finger2_link2").id]).name == "r_index_finger_mcp_abd"
    for source in ("rh_thmiddle", "rh_thdistal"):
        record = by_source[source]
        assert record["semantic_region"] == "palmar"
        assert record["normal_error_deg"] < 20
        np.testing.assert_allclose(
            np.dot(record["source_chart_normal"], record["target_chart_normal"]),
            np.cos(np.deg2rad(record["normal_error_deg"])), atol=1e-7,
        )
    np.testing.assert_allclose(by_source["rh_thmiddle"]["source_frame"]["palmar"], [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(by_source["rh_thmiddle"]["target_frame"]["palmar"], [0, 1, 0], atol=1e-7)


def test_palm_patch_uses_exterior_skin_and_landmarks_instead_of_body_zero_plane(wuji):
    _, (_, _, _, records, masks) = _match(wuji, "1_Large_Diameter")
    index = next(i for i, record in enumerate(records) if record["shadow_body"] == "rh_palm")
    record, patch = records[index], masks[index]
    # The distal wrist's exposed palm can face +Y while lying at negative Y.
    # A body-origin half-space would incorrectly reject this real skin.
    with np.load(query_path(wuji, "right_palm_link")) as pool:
        palmar = np.asarray(record["target_frame"]["palmar"])
        front_at_negative_y = ((pool["local_pos"] @ palmar < 0)
                               & (pool["local_normal"] @ palmar > .7))
        assert np.any(patch["mask"] & front_at_negative_y)
    assert abs(record["source_normalized_coordinates"][2]
               - record["target_normalized_coordinates"][2]) < .1
    assert "shell depth" in record["palm_depth_policy"]


def test_tip_cap_and_side_annotations_keep_distinct_patch_identity(wuji):
    _, (_, _, _, records, _) = _match(wuji, "24_Tip_Pinch")
    assert all(record["semantic_region"] == "tip" for record in records)
    assert all(record["longitudinal_window"] == [.75, 1.35] for record in records)
    _, (_, _, _, lateral, _) = _match(wuji, "16_Lateral")
    assert any(record["semantic_region"].startswith("lateral") for record in lateral)


def test_empty_semantic_patch_reports_failure_without_link_fallback(wuji, monkeypatch):
    real = patches._target_pool

    def empty(*args, **kwargs):
        pool = real(*args, **kwargs)
        return {**pool, "base": np.zeros_like(pool["base"])}

    monkeypatch.setattr(patches, "_target_pool", empty)
    with pytest.raises(ValueError, match="Empty semantic patch"):
        _match(wuji, "1_Large_Diameter")


def test_all_available_templates_have_nonempty_semantic_correspondences(wuji):
    templates = sorted((TEMPLATES / "assets/hand/shadow/raw_anno").glob("*.yaml"))
    assert len(templates) >= 33
    count = 0
    for path in templates:
        contacts, (_, _, _, records, masks) = _match(wuji, path.stem)
        count += len(contacts)
        assert len(records) == len(contacts), path.stem
        assert all(patch["mask"].any() for patch in masks), path.stem
    assert count >= 250


@pytest.mark.parametrize("name", ["6_Prismatic_4_Finger", "7_Prismatic_3_Finger", "8_Prismatic_2_Finger"])
def test_central_tip_policy_never_overrides_prismatic_semantic_patch(wuji, name):
    from types import SimpleNamespace
    from retarget.env.grasp.tip_patch import centered_template_contacts, configure_tip_centers
    shadow, _, contacts, _ = load_template(TEMPLATES, name)
    contacts, _ = centered_template_contacts(TEMPLATES, name, contacts)
    bodies, local, normals, records, regions = patches.match_semantic_patches(shadow, contacts, wuji)
    masks = [region["mask"].copy() for region in regions]
    before = local.copy()
    hand = SimpleNamespace(model=wuji, bodies=bodies, local=local.copy(), normals=normals.copy())
    configure_tip_centers(hand, name, records, semantic_masks=masks)
    assert any("central_tip_override_skipped" in record for record in records)
    for i, (body, record) in enumerate(zip(bodies, records)):
        assert masks[i][record["query_index"]]
        if "central_tip_override_skipped" in record:
            np.testing.assert_array_equal(hand.local[i], before[i])
            assert int(body) not in hand.tip_patches
        else:
            assert hand.tip_patches[int(body)]["mask"][record["query_index"]]
        np.testing.assert_array_equal(masks[i], regions[i]["mask"])
