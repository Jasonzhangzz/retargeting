"""Real sample integrity and camera-pose conversion, including invalid frames."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]/'data/dexycb'
ITEMS=json.loads((ROOT/'index.json').read_text())


def test_five_distinct_grasped_objects():
    assert len(ITEMS)==5
    assert len({x['object'] for x in ITEMS})==5


@pytest.mark.parametrize('item',ITEMS,ids=[x['object'] for x in ITEMS])
def test_sample_integrity_and_original_object_transforms(item):
    path=ROOT/item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    meta=json.loads((path.parent/'meta.json').read_text())
    source=yaml.safe_load((path.parent/'source_meta.yml').read_text())
    with np.load(path,allow_pickle=False) as a:
        n=item['frames'];valid=a['frame_valid']
        assert a['hand_pose_51'].shape==(n,51)
        assert a['hand_shape_10'].shape==(10,)
        assert a['joint_3d_camera'].shape==(n,21,3)
        assert valid.shape==(n,) and valid.dtype==bool
        assert valid.sum()==item['valid_frames']
        np.testing.assert_array_equal(a['frame_indices'],np.arange(n))
        np.testing.assert_allclose(a['times'],np.arange(n)/30.,rtol=0,atol=0)
        assert np.isfinite(a['hand_pose_51']).all()
        assert np.isfinite(a['hand_shape_10']).all()
        transform=a['object_transform_camera'][:,source['ycb_grasp_ind']]
        pose=a['object_pose_camera_7']
        np.testing.assert_allclose(pose[:,:4][valid],Rotation.from_matrix(transform[valid,:,:3]).as_quat(),atol=1.e-7)
        np.testing.assert_array_equal(pose[:,4:],transform[:,:,3])
        assert meta['sequence']==path.parent.name and meta['object']==item['object']
        assert meta['license']=='CC-BY-NC-4.0'
