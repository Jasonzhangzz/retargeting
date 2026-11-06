#!/usr/bin/env python3
"""Freeze the saved local hand as a mirrored right MANO asset for grasp CEM."""
import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--shadow-xml', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'MANIFEST.json').exists():
        raise FileExistsError('Use a fresh immutable model export')
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from scripts.audit_moxian_calibration import restore
    from scripts.inspect_shadow_mano_mapping import map_contacts
    from retarget.env.grasp.mano_solver import ManoDecoder
    import mujoco

    torch.set_num_threads(1)
    profile_bytes = args.profile.read_bytes()
    profile = json.loads(profile_bytes)
    if profile['side'] != 'left' or not profile['calibrated']:
        raise ValueError('Expected the saved calibrated local left hand')
    (out / 'profile.json').write_bytes(profile_bytes)
    surface = ManoSurface(DEFAULT_MANO)
    restore(surface, profile)
    b = surface.model
    reflect = np.array([-1., 1., 1.])
    axial = np.array([1., -1., -1.])
    js = (ROOT / 'moxian_web/static/mano-view.js').read_text()
    skin = re.search(r"color:'(#[0-9a-fA-F]{6})',roughness:([\d.]+),metalness:([\d.]+)", js)
    if skin is None:
        raise ValueError('Cannot identify the current local MANO material')
    appearance = dict(skin=skin[1], roughness=float(skin[2]), metalness=float(skin[3]))
    metadata = dict(source_profile_revision=profile['revision'], source_geometry_version=surface.shape_version,
                    source_profile_sha256=hashlib.sha256(profile_bytes).hexdigest(), source_side='left', side='right',
                    mirrored=True, units='m', fixed_calibrated_geometry=True, pose_correctives=False,
                    dimensions_mm=surface.dimension_summary(), solver_pose_representation='MANO PCA')
    cpu = lambda x: x.detach().cpu().numpy()
    data = dict(v_template=np.asarray(surface.rest_np) * reflect,
                retarget_rest_joints=np.asarray(surface.rest_j_np) * reflect,
                weights=cpu(b.weights), J_regressor=cpu(b.regressor),
                shapedirs=np.zeros_like(cpu(b.shapedirs)), posedirs=np.zeros_like(cpu(b.posedirs)),
                hands_components=(cpu(b.components).reshape(-1, 15, 3) * axial).reshape(-1, 45),
                hands_mean=(cpu(b.pose_mean).reshape(15, 3) * axial).reshape(45),
                f=np.asarray(surface.faces)[:, [0, 2, 1]],
                kintree_table=np.array([[2**32-1, *b.parents[1:]], list(range(16))], dtype=np.uint32),
                retarget_appearance=appearance, retarget_metadata=metadata)
    model = out / 'MANO_RIGHT.pkl'
    with model.open('wb') as stream:
        pickle.dump(data, stream, protocol=4)
    np.savez_compressed(out / 'rest_geometry.npz', vertices=data['v_template'], joints=data['retarget_rest_joints'], faces=data['f'])
    (out / 'appearance.json').write_text(json.dumps(appearance, indent=2) + '\n')
    decoder = ManoDecoder(model, 'cpu')
    np.testing.assert_allclose(cpu(decoder.rest), data['v_template'], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(cpu(decoder.joints_rest), data['retarget_rest_joints'], atol=1e-8, rtol=1e-6)
    # Decode with exactly zero local rotations to check rest reconstruction.
    z = torch.zeros(10, dtype=decoder.rest.dtype)
    vertices, joints, _ = decoder.base(z, z.new_zeros(()), z[:5], z[:5], z.new_zeros(1, 15, 3))
    np.testing.assert_allclose(cpu(vertices[0]), data['v_template'], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(cpu(joints[0]), data['retarget_rest_joints'], atol=1e-7, rtol=1e-5)
    old = json.loads(args.mapping.read_text())
    native = mujoco.MjModel.from_xml_path(str(args.shadow_xml))
    mappings = {}
    for typ, records in old['mappings'].items():
        contacts = [(r['shadow_body'], np.array(r['shadow_local_point'] + r['shadow_local_normal'])) for r in records]
        mappings[typ] = map_contacts(native, contacts, decoder)
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    (out / 'fixed_contact_mapping.json').write_text(json.dumps(dict(model_sha256=model_hash, mappings=mappings), indent=2) + '\n')
    # Exercise the same anatomical patch construction used by the production adapter.
    from retarget.env.grasp.mano_patch_anchor import patch_faces
    from retarget.solver.cem import DEFAULTS
    for records in mappings.values():
        assert all(patch_faces(decoder, records, DEFAULTS.patch.patch_radius_m))
    report = dict(**metadata, appearance=appearance, model_sha256=model_hash, types=len(mappings),
                  slots=sum(map(len, mappings.values())), rest_geometry_verified=True,
                  sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out / 'MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
