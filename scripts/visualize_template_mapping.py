#!/usr/bin/env python3
"""Export an offline, paired Shadow -> Wuji2 template inspection page.

Uses the production fixed-template bone conversion and contact mapping. No
object-dependent refinement is applied to these poses.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from plotly.offline import get_plotlyjs

from retarget.env.grasp.hand import hand_tree, query_path
from retarget.env.grasp.graspadmm_inputs import fixed_finger_angles
from retarget.env.grasp.inputs import adapt_primitive_contacts
from retarget.env.grasp.contact_regions import adapt_contact_regions
from retarget.task.grasp.dexonomy import load_template, map_contacts, anatomical_joints
from retarget.initialization.ik import palm_frame


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(model, data, shadow):
    """A common, anatomically aligned view, fixed across posed/neutral modes."""
    neutral = mujoco.MjData(model)
    neutral.qpos[:] = 0.
    mujoco.mj_kinematics(model, neutral)
    if shadow:
        joints = anatomical_joints(model, neutral)
        origin, basis = palm_frame(joints[0], joints[[1, 5, 9, 13, 17]])
    else:
        roots = [neutral.xpos[model.body(f'right_finger{i}_link2').id] for i in range(1, 6)]
        origin, basis = palm_frame(neutral.xpos[model.body('right_palm_link').id], np.asarray(roots))
    # palm_frame columns: lateral, longitudinal, palmar; display Z=longitudinal.
    basis = basis[:, [0, 2, 1]]
    # This reordering reverses handedness. Reverse lateral too to preserve SO(3).
    basis[:, 0] *= -1
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(basis), 1., atol=1e-10)
    return origin, basis


def visual_assets(model):
    meshes = {}
    for mesh in range(model.nmesh):
        v, n = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        f, k = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        meshes[str(mesh)] = dict(v=model.mesh_vert[v:v+n].round(7).tolist(),
                                 f=model.mesh_face[f:f+k].tolist())
    return meshes


def pose(model, data, origin, basis):
    geoms, seen = [], set()
    # Render visual meshes once, without overlapping collision copies.
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body, mesh = int(model.geom_bodyid[geom]), int(model.geom_dataid[geom])
        key = (body, mesh, *model.geom_pos[geom], *model.geom_quat[geom])
        if key in seen:
            continue
        seen.add(key)
        geoms.append(dict(mesh=str(mesh), body=model.body(body).name,
                          p=((data.geom_xpos[geom]-origin) @ basis).tolist(),
                          r=(basis.T @ data.geom_xmat[geom].reshape(3, 3)).tolist()))
    if model.body(1).name == 'rh_palm':
        joints = anatomical_joints(model, data)
    else:
        joints = [data.xpos[model.body('right_palm_link').id]]
        for i, name in enumerate(('thumb', 'index', 'middle', 'ring', 'pinky'), 1):
            joints.extend(data.xpos[model.body(f'right_finger{i}_link{k}').id] for k in (2, 3, 4))
            joints.append(data.site_xpos[model.site(f'right_{name}_tip').id])
        joints = np.asarray(joints)
    return dict(geoms=geoms, joints=((joints-origin) @ basis).tolist())


def world_contacts(model, data, names, local, normals, origin, basis):
    ids = np.asarray([model.body(name).id for name in names])
    rotations = data.xmat[ids].reshape(-1, 3, 3)
    p = data.xpos[ids] + np.einsum('nij,nj->ni', rotations, local)
    n = np.einsum('nij,nj->ni', rotations, normals)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return dict(p=((p-origin) @ basis).tolist(), n=(n @ basis).tolist())


def export(args):
    target = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'), encoding='unicode'))
    shadow_path = args.checkout / 'assets/hand/shadow/right.xml'
    tree = ET.parse(shadow_path).getroot()
    tree.find('compiler').set('meshdir', str(args.shadow_meshes.resolve()))
    visual = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(meshes=dict(shadow=visual_assets(visual), wuji=visual_assets(target)), cases=[])
    conversion_args = SimpleNamespace(dexonomy_root=args.checkout, init_device=args.device)
    annotations = sorted((args.checkout/'assets/hand/shadow/raw_anno').glob('*.yaml'),
                         key=lambda p: (int(p.stem.partition('_')[0]) if p.stem.partition('_')[0].isdigit() else 100, p.stem))
    if args.templates:
        annotations = [p for p in annotations if p.stem in args.templates]
    reports = []
    for annotation in annotations:
        name = annotation.stem
        shadow, source, original, _ = load_template(args.checkout, name)
        contacts, adaptations = adapt_primitive_contacts(args.checkout, name, original)
        bodies, local, normals, direct = map_contacts(shadow, contacts, target)
        raw_local, raw_normals = local.copy(), normals.copy()
        effective = json.loads(json.dumps(direct))
        local, normals, effective, region = adapt_contact_regions(target, name, bodies, local, normals, effective)
        fingers, conversion = fixed_finger_angles(target, anatomical_joints(shadow, source)[None],
                                                 conversion_args, annotation,
                                                 cache_root=out/'template_cache')
        sd, wd = mujoco.MjData(visual), mujoco.MjData(target)
        so, sb = canonical(visual, sd, True)
        wo, wb = canonical(target, wd, False)
        names = [target.body(int(b)).name for b in bodies]
        source_names = [b for b, _ in contacts]
        source_values = np.asarray([v for _, v in contacts])
        case = dict(name=name, original_contact_count=len(original), adaptations=adaptations,
                    region=region, mapping=dict(direct=direct, effective=effective),
                    source_annotation=str(annotation.relative_to(ROOT)), annotation_sha256=digest(annotation),
                    finger_qpos=fingers.tolist(), shadow_qpos=source.qpos.tolist(),
                    conversion=conversion, poses={})
        for mode in ('grasp', 'neutral'):
            sd.qpos[:] = source.qpos if mode == 'grasp' else 0.
            wd.qpos[:] = np.r_[np.zeros(6), fingers] if mode == 'grasp' else 0.
            mujoco.mj_kinematics(visual, sd)
            mujoco.mj_kinematics(target, wd)
            s, w = pose(visual, sd, so, sb), pose(target, wd, wo, wb)
            s['contacts'] = world_contacts(visual, sd, source_names, source_values[:, :3], source_values[:, 3:], so, sb)
            w['direct'] = world_contacts(target, wd, names, raw_local, raw_normals, wo, wb)
            w['effective'] = world_contacts(target, wd, names, local, normals, wo, wb)
            # Per-bone angular discrepancy in the same neutral anatomical frames.
            a = np.diff(np.asarray(s['joints'])[1:].reshape(5, 4, 3), axis=1)
            b = np.diff(np.asarray(w['joints'])[1:].reshape(5, 4, 3), axis=1)
            cosine = np.sum(a*b, -1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
            angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            case['poses'][mode] = dict(shadow=s, wuji=w, bone_angles_deg=angles.tolist())
        neutral = case['poses']['neutral']
        for mode, records in case['mapping'].items():
            cosine = np.sum(np.asarray(neutral['shadow']['contacts']['n']) *
                            np.asarray(neutral['wuji'][mode]['n']), axis=-1)
            for record, angle in zip(records, np.degrees(np.arccos(np.clip(cosine, -1., 1.)))):
                record['neutral_normal_deviation_deg'] = float(angle)
        errors = [v['mapping_distance_mm'] for v in direct]
        case['max_mapping_distance_mm'] = max(errors)
        case['max_bone_angle_deg'] = float(np.max(case['poses']['grasp']['bone_angles_deg']))
        # Verify exported anchors against the exact saved query array/order.
        for records in (direct, effective):
            for record in records:
                with np.load(query_path(target, record['wuji_body'])) as pool:
                    np.testing.assert_allclose(pool['local_pos'][record['query_index']], record['wuji_local'], atol=1e-9)
        reports.append(dict(template=name, contacts=len(contacts),
                            max_mapping_distance_mm=max(errors),
                            max_bone_angle_deg=case['max_bone_angle_deg'],
                            max_direct_neutral_normal_deviation_deg=max(r['neutral_normal_deviation_deg'] for r in direct)))
        payload['cases'].append(case)
        print(f'{name}: {len(contacts)} contacts, max mapping {max(errors):.2f} mm', flush=True)
    if not payload['cases']:
        raise ValueError('No templates selected')
    revision_file = args.shadow_meshes/'revision.json'
    payload['provenance'] = dict(shadow_mesh_commit=json.loads(revision_file.read_text())['sha'] if revision_file.exists() else None,
                                 shadow_xml_sha256=digest(shadow_path),
                                 wuji_xml_sha256=digest(ROOT/'data/grasp/assets/wuji2/right.xml'),
                                 policy='Object-independent production bone_seed_cuda + map_contacts; effective mode additionally uses adapt_primitive_contacts and adapt_contact_regions. No solver output.')
    (out/'data.json').write_text(json.dumps(payload, separators=(',', ':')))
    (out/'audit.json').write_text(json.dumps(dict(cases=len(reports), query_anchor_checks='passed', cases_summary=reports, provenance=payload['provenance']), indent=2))
    (out/'plotly.min.js').write_text(get_plotlyjs())
    template = (ROOT/'retarget/viewer/template_mapping.html').read_text()
    (out/'index.html').write_text(template.replace('/*__DATA__*/null', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')))
    print(f'Open {out / "index.html"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, default=ROOT/'data/grasp/templates')
    parser.add_argument('--shadow-meshes', type=Path, default=ROOT/'.cache/shadow_hand_visual')
    parser.add_argument('--output', type=Path, default=ROOT/'output/template_mapping')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--templates', nargs='+')
    export(parser.parse_args())
