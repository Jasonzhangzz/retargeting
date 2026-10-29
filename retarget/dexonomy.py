"""Dexonomy Shadow annotations -> Wuji surface queries -> object SDF anchors.

Only the external checkout's YAML and kinematic XML are read. No Dexonomy
Python dependencies, pickled templates, or Shadow mesh assets are required.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from retarget.mano_noise import nearest_sdf, palm_frame, retarget_from_mano

ROOT = Path(__file__).resolve().parents[1]
FINGERS = {'th': (1, 1), 'ff': (2, 5), 'mf': (3, 9),
           'rf': (4, 13), 'lf': (5, 17)}


def shadow_kinematics(path):
    """Compile exactly the annotated joint tree without external visual assets."""
    tree = ET.parse(path).getroot()
    for parent in tree.iter():
        for child in list(parent):
            if child.tag in ('asset', 'geom', 'site', 'actuator', 'tendon',
                             'contact', 'sensor', 'keyframe'):
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))


def annotation_entries(value):
    """An integer, a six-vector, or a mixed list of either (upstream format)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list):
        if len(value) == 6 and all(isinstance(v, (int, float)) for v in value):
            return [value]
        return value
    raise ValueError(f'Invalid Dexonomy contact annotation: {value!r}')


def load_template(checkout, name):
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Dexonomy template must be a single name, without a path')
    hand = Path(checkout).expanduser().resolve() / 'assets/hand/shadow'
    path = hand / 'raw_anno' / f'{name}.yaml'
    annotation = yaml.safe_load(path.read_text())
    keypoints = yaml.safe_load((hand / 'keypoint.yaml').read_text())
    model = shadow_kinematics(hand / 'right.xml')
    pose = np.asarray(annotation['qpos'], float)
    if pose.shape != (model.nq,) or not np.isfinite(pose).all():
        raise ValueError(f'Invalid Shadow qpos in {path}: {pose.shape}')
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    contacts = []
    for body, values in (annotation.get('contact') or {}).items():
        body_id = model.body(body).id
        rotation = data.xmat[body_id].reshape(3, 3)
        for entry in annotation_entries(values):
            if isinstance(entry, int) and not isinstance(entry, bool):
                resolved = body
                visited = set()
                while isinstance(keypoints[resolved], str):
                    if resolved in visited:
                        raise ValueError(f'Cyclic keypoint alias: {body}')
                    visited.add(resolved)
                    resolved = keypoints[resolved]
                if entry < 0 or entry >= len(keypoints[resolved]):
                    raise ValueError(f'Invalid keypoint {entry} for {body}')
                local = np.asarray(keypoints[resolved][entry], float)
            else:
                world = np.asarray(entry, float)
                if world.shape != (6,):
                    raise ValueError(f'Contact for {body} must contain six numbers')
                local = np.r_[(world[:3] - data.xpos[body_id]) @ rotation,
                              world[3:] @ rotation]
            if local.shape != (6,) or not np.isfinite(local).all() or np.linalg.norm(local[3:]) < 1e-8:
                raise ValueError(f'Invalid contact position/normal for {body}')
            contacts.append((body, local))
    if not contacts:
        raise ValueError(f'Template has no annotated contacts: {path}')
    return model, data, contacts, path


def anatomical_joints(model, data):
    """Export Shadow bones in the wrist + five four-point chain convention."""
    joints = np.empty((21, 3))
    joints[0] = data.xpos[model.body('rh_palm').id]
    for prefix, (_, start) in FINGERS.items():
        ids = [model.body(f'rh_{prefix}{part}').id
               for part in ('proximal', 'middle', 'distal')]
        joints[start:start+3] = data.xpos[ids]
        distal = ids[-1]
        joints[start+3] = data.xpos[distal] + data.xmat[distal].reshape(3, 3) @ [0., 0., .027]
    return joints


def wuji_body(source):
    if source == 'rh_palm':
        return 'right_palm_link'
    for prefix, (finger, _) in FINGERS.items():
        for part, link in (('proximal', 2), ('middle', 3), ('distal', 4)):
            if source == f'rh_{prefix}{part}':
                return f'right_finger{finger}_link{link}'
    # Wuji has no independently articulated little-finger metacarpal.
    if source == 'rh_lfmetacarpal':
        return 'right_palm_link'
    raise ValueError(f'No Shadow -> Wuji semantic mapping for {source}')


def map_contacts(shadow, contacts, wuji):
    """Map neutral anatomical coordinates, then select an actual Wuji query.

    Joint-local axes differ across the hands. Align neutral palm frames first,
    scale each bone by its length, and use position and normal together when
    selecting a point on the corresponding Wuji link's query surface.
    """
    from retarget.grasp_hand import query_path
    sd, wd = mujoco.MjData(shadow), mujoco.MjData(wuji)
    mujoco.mj_kinematics(shadow, sd)
    # Use true zero neutral geometry, even for the thumb's positive lower bound.
    wd.qpos[:26] = 0.
    mujoco.mj_kinematics(wuji, wd)
    sj = anatomical_joints(shadow, sd)
    so, sb = palm_frame(sj[0], sj[[1, 5, 9, 13, 17]])
    roots = np.array([wd.xpos[wuji.body(f'right_finger{i}_link2').id] for i in range(1, 6)])
    wo, wb = palm_frame(wd.xpos[wuji.body('right_palm_link').id], roots)
    alignment = wb @ sb.T
    points, normals, bodies, records = [], [], [], []
    for source, local in contacts:
        target = wuji_body(source)
        si, wi = shadow.body(source).id, wuji.body(target).id
        sr, wr = sd.xmat[si].reshape(3, 3), wd.xmat[wi].reshape(3, 3)
        transform = wr.T @ alignment @ sr
        if target == 'right_palm_link':
            # Normalize around the anatomical palm center for metacarpal too.
            desired = ((sd.xpos[si] + sr @ local[:3] - so) @ alignment.T
                       + wo - wd.xpos[wi]) @ wr
        else:
            link = int(target[-1])
            if link < 4:
                next_body = wuji.body(target[:-1] + str(link+1)).id
                target_length = np.linalg.norm(wd.xpos[next_body] - wd.xpos[wi])
                child = np.flatnonzero(shadow.body_parentid == si)[0]
                source_length = np.linalg.norm(sd.xpos[child] - sd.xpos[si])
            else:
                finger = int(target[len('right_finger')])
                finger_name = ('thumb', 'index', 'middle', 'ring', 'pinky')[finger-1]
                target_length = np.linalg.norm(wd.site_xpos[wuji.site(f'right_{finger_name}_tip').id] - wd.xpos[wi])
                source_length = .027
            desired = transform @ local[:3] * (target_length / source_length)
        normal = transform @ local[3:]
        normal /= np.linalg.norm(normal)
        with np.load(query_path(wuji, target)) as queries:
            qp, qn = queries['local_pos'], queries['local_normal']
        qn = qn / np.maximum(np.linalg.norm(qn, axis=1, keepdims=True), 1e-12)
        score = np.sum((qp-desired)**2, axis=1) + .01**2 * (1.-qn @ normal)
        selected = int(np.argmin(score))
        points.append(qp[selected]); normals.append(qn[selected]); bodies.append(wi)
        records.append(dict(shadow_body=source, wuji_body=target,
                            shadow_local=local.tolist(), query_index=selected,
                            wuji_local=qp[selected].tolist(),
                            wuji_local_normal=qn[selected].tolist(),
                            mapping_distance_mm=float(np.linalg.norm(qp[selected]-desired)*1000)))
    return np.asarray(bodies), np.asarray(points), np.asarray(normals), records
