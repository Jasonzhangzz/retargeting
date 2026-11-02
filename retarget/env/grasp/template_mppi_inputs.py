"""Immutable direct Shadow-to-Wuji templates and root-only pose initialization."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget.config.run import ROOT
from retarget.task.grasp.dexonomy import anatomical_joints, load_template, map_contacts, wuji_body
from .graspadmm_scene import required_body_groups
from .hand import asset_root, model_hand, query_path


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _immutable_array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    # A bytes-backed array cannot have writeability re-enabled by its consumer.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class TemplateContacts:
    body_names: tuple[str, ...]
    local_points: np.ndarray
    local_normals: np.ndarray
    finger_qpos: np.ndarray
    required_cbody: tuple[tuple[str, ...], ...]
    provenance: dict

    def __post_init__(self):
        names = tuple(self.body_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Template requires nonempty contact body names")
        groups = tuple(tuple(group) for group in self.required_cbody)
        if not groups or any(not group or set(group) - set(names) for group in groups):
            raise ValueError("Required contact groups must refer to annotated bodies")
        points = _immutable_array(self.local_points, (len(names), 3), "local_points")
        normals = _immutable_array(self.local_normals, (len(names), 3), "local_normals")
        if not np.allclose(np.linalg.norm(normals, axis=-1), 1., rtol=1.e-6, atol=1.e-6):
            raise ValueError("Template local normals must be unit vectors")
        object.__setattr__(self, "body_names", names)
        object.__setattr__(self, "local_points", points)
        object.__setattr__(self, "local_normals", normals)
        object.__setattr__(self, "finger_qpos", _immutable_array(self.finger_qpos, (20,), "finger_qpos"))
        object.__setattr__(self, "required_cbody", groups)
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))

    def to_dict(self):
        return dict(body_names=list(self.body_names), local_points=self.local_points.tolist(),
                    local_normals=self.local_normals.tolist(), finger_qpos=self.finger_qpos.tolist(),
                    required_cbody=[list(group) for group in self.required_cbody],
                    provenance=_plain(self.provenance))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked_finger_qpos(template_name, checkout, annotation_path):
    """Reuse only the bone pose of the inspected mapping artifact, never its contacts."""
    path = ROOT / "data/grasp/assets/wuji2/template_bone_poses.json"
    if not path.is_file():
        path = ROOT / "output/template_mapping/data.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    record = next((row for row in data.get("cases", ()) if row.get("name") == template_name), None)
    if record is None:
        return None
    expected = data.get("provenance", {})
    checks = dict(annotation=record.get("annotation_sha256") == _sha(annotation_path),
                  shadow_xml=expected.get("shadow_xml_sha256") == _sha(checkout / "assets/hand/shadow/right.xml"),
                  wuji_xml=expected.get("wuji_xml_sha256") == _sha(asset_root("wuji2") / "right.xml"))
    if not all(checks.values()):
        return None
    conversion = record.get("conversion", {})
    if (conversion.get("method") != "CUDA_batched_finger_bone_direction_LM"
            or conversion.get("object_or_contact_objective_used") is not False):
        return None
    q = np.asarray(record.get("finger_qpos"), dtype=np.float64)
    if q.shape != (20,) or not np.isfinite(q).all():
        raise ValueError("Checked template artifact contains invalid finger coordinates")
    return q, dict(method="checked_object_independent_bone_pose", path=str(path),
                   sha256=_sha(path), source_hash_checks=checks, conversion=conversion)


def load_template_contacts(template_name, dexonomy_root, model, *, device="cuda:0", cache_root=None):
    """Map the original annotation slots exactly once; no contact or object adapters."""
    checkout = Path(dexonomy_root).expanduser().resolve()
    if model_hand(model) != "wuji2":
        raise ValueError("Direct template MPPI requires Wuji2")
    shadow, posed, contacts, annotation_path = load_template(checkout, template_name)
    bodies, points, normals, mapping = map_contacts(shadow, contacts, model)
    names = tuple(model.body(int(body)).name for body in bodies)
    checked = _checked_finger_qpos(template_name, checkout, annotation_path)
    if checked is None:
        from .graspadmm_inputs import fixed_finger_angles

        args = SimpleNamespace(dexonomy_root=checkout, init_device=device, gesture_policy="legacy")
        fingers, conversion = fixed_finger_angles(model, anatomical_joints(shadow, posed)[None],
                                                  args, annotation_path, cache_root=cache_root)
    else:
        fingers, conversion = checked
    import yaml

    annotation = yaml.safe_load(annotation_path.read_text())
    if annotation.get("required_cbody") is None:
        required = required_body_groups(names)
    else:
        required = [list(dict.fromkeys(wuji_body(name) for name in group))
                    for group in annotation["required_cbody"]]
    provenance = dict(template_name=template_name, policy="original_annotation_direct_mapping_bone_pose",
                      annotation=str(annotation_path), annotation_sha256=_sha(annotation_path),
                      keypoints=str(checkout / "assets/hand/shadow/keypoint.yaml"),
                      keypoints_sha256=_sha(checkout / "assets/hand/shadow/keypoint.yaml"),
                      shadow_xml_sha256=_sha(checkout / "assets/hand/shadow/right.xml"),
                      wuji_xml_sha256=_sha(asset_root("wuji2") / "right.xml"),
                      query_sha256={name: _sha(query_path(model, name)) for name in dict.fromkeys(names)},
                      mapping=mapping, pose_conversion=conversion, original_contact_count=len(contacts),
                      primitive_contact_adaptations=False, region_adaptations=False,
                      object_dependent_fitting=False)
    return TemplateContacts(names, points, normals, fingers, tuple(map(tuple, required)), provenance)


def load_solver_template_contacts(template_name, dexonomy_root, model, **kwargs):
    """Apply the explicit pad-center policy once, before freezing solver slots.

    The direct annotation loader remains available for source audits. Types 6/7
    use a new fixed template; Type 8 and all other types retain direct mapping.
    No point is subsequently selected or moved during IK/MPPI.
    """
    template = load_template_contacts(template_name, dexonomy_root, model, **kwargs)
    if template_name.partition('_')[0] not in ('6', '7'):
        return template
    from .contact_regions import adapt_contact_regions
    bodies = np.array([model.body(name).id for name in template.body_names])
    provenance = _plain(template.provenance)
    points, normals, mapping, region = adapt_contact_regions(model, template_name,
        bodies, template.local_points, template.local_normals, provenance['mapping'])
    provenance.update(policy='fixed_prismatic_pad_center_v1', region_adaptations=True,
        original_mapped_contacts=dict(body_names=list(template.body_names),
            local_points=template.local_points.tolist(),local_normals=template.local_normals.tolist()),
        mapping=mapping, fixed_contact_region=region,
        contact_policy_authorization='User requested Type 6/7 finger-pad centers; Type 8 unchanged',
        contacts_fixed_before_object_fitting=True)
    return TemplateContacts(template.body_names, points, normals, template.finger_qpos,
                            template.required_cbody, provenance)


def _root_matcher_modules():
    from retarget.initialization.dexonomy.op import gen_init
    from retarget.initialization.dexonomy.data import obj_loader, hand_loader
    from retarget.initialization.dexonomy.util import warp_util

    return gen_init, obj_loader, hand_loader, warp_util


def canonical_template_contacts(model, templates, *, device):
    """One CUDA FK batch for all fixed canonical templates, including points/normals."""
    import torch
    from .hand import HAND_BODIES
    from .template import WujiContactKinematics
    from .template_mppi_reference import template_signature

    if not templates:
        return []
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Canonical template batch FK requires CUDA")
    fk = WujiContactKinematics(model, [model.body(name).id for name in HAND_BODIES], target)
    count, slots = len(templates), max(len(t.body_names) for t in templates)
    ids = np.zeros((count, slots), int)
    local = np.zeros((count, slots, 3))
    normals = np.zeros_like(local)
    for row, template in enumerate(templates):
        n = len(template.body_names)
        ids[row, :n] = [HAND_BODIES.index(name) for name in template.body_names]
        local[row, :n], normals[row, :n] = template.local_points, template.local_normals
    with torch.no_grad():
        q = fk.tensor(np.array([np.r_[np.zeros(6), t.finger_qpos] for t in templates]))
        p, r = torch.vmap(fk.frames)(q)
        rows = torch.arange(count, device=target)[:, None]
        columns = torch.as_tensor(ids, device=target)
        p, r = p[rows, columns], r[rows, columns]
        world = p + torch.einsum("bnij,bnj->bni", r, fk.tensor(local))
        normal = torch.einsum("bnij,bnj->bni", r, fk.tensor(normals))
        contacts = torch.cat((world, normal), -1).float().cpu().numpy()
    return [dict(contacts=contacts[i, :len(t.body_names)].copy(),
                 template_signature=template_signature(t),
                 backend="torch_cuda_batched_WujiContactKinematics", batch_size=count)
            for i, t in enumerate(templates)]


def initialize_root_candidates(hand, template, *, seed, device, points=24, inplane=8, iterations=60,
                               canonical_contacts=None):
    """Fit only root SE(3) with the retained matcher; retain every finite proposal."""
    import torch
    from .initialization import matching_mesh, sample_original_poses

    for key, value in dict(seed=seed, points=points, inplane=inplane, iterations=iterations).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a {'nonnegative' if key == 'seed' else 'positive'} integer")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Root-only native mesh matching requires an available CUDA device")
    bodies = np.array([hand.model.body(name).id for name in template.body_names])
    if (not np.array_equal(hand.bodies, bodies)
            or not np.array_equal(hand.local, template.local_points)
            or not np.array_equal(hand.normals, template.local_normals)):
        raise ValueError("StaticHand must retain the exact immutable template slots")
    gen_init, obj_loader, hand_loader, warp_util = _root_matcher_modules()
    original_q = hand.data.qpos.copy()
    if target_device.type == "cuda":
        from .template_mppi_reference import template_signature
        if canonical_contacts is None:
            canonical_contacts = canonical_template_contacts(hand.model, [template], device=target_device)[0]
        if canonical_contacts["template_signature"] != template_signature(template):
            raise ValueError("Canonical GPU FK contacts do not match the fixed template")
        cpn = np.asarray(canonical_contacts["contacts"], np.float32)
        if cpn.shape != (len(template.body_names), 6) or not np.isfinite(cpn).all():
            raise ValueError("Invalid canonical GPU FK points/normals")
        fk_metadata = dict(backend=canonical_contacts["backend"], batch_size=canonical_contacts["batch_size"])
    else:
        if canonical_contacts is not None:
            raise ValueError("Precomputed CUDA FK contacts cannot enter the CPU reference path")
        hand.set(np.r_[np.zeros(6), template.finger_qpos])
        world = hand.points(bodies, template.local_points)
        normal = np.einsum("nij,nj->ni", hand.data.xmat[bodies].reshape(-1, 3, 3), template.local_normals)
        cpn = np.c_[world, normal].astype(np.float32)
        fk_metadata = dict(backend="mujoco_cpu_reference", batch_size=1)
    payload = dict(grasp_qpos=np.r_[np.zeros(3), 1., np.zeros(3), template.finger_qpos].astype(np.float32),
                   hand_cpn_w=cpn, n_evo=np.int64(0))
    loader = object.__new__(hand_loader.HandTemplateLoader)
    loader.skt_data, loader.skt_bid = np.empty((0, 6), np.float32), np.empty(0, int)

    def forward_kinematics(finger_q):
        if target_device.type == "cuda":
            # The retained loader only requests skeleton transforms, and this
            # fixed-contact path has no skeleton. Its contact FK was batched on
            # CUDA above. Never redo that FK on CPU just to index empty arrays.
            if not np.array_equal(finger_q, template.finger_qpos.astype(np.float32)):
                raise ValueError("The fixed template fingers changed in the matcher loader")
            return np.empty((0, 3, 3), np.float32), np.empty((0, 3), np.float32)
        hand.set(np.r_[np.zeros(6), finger_q])
        return hand.data.xmat.reshape(-1, 3, 3).astype(np.float32), hand.data.xpos.astype(np.float32)

    # The retained contact-frame loader reads a path; this temporary payload is
    # generated solely from our trusted arrays and never changes the template.
    previous_rng = np.random.get_state()
    try:
        np.random.seed(seed)
        with tempfile.TemporaryDirectory(prefix="template-mppi-root-") as directory:
            path = Path(directory) / "template.npy"
            np.save(path, payload)
            prepared = loader._load_data(SimpleNamespace(forward_kinematics=forward_kinematics), str(path))
    finally:
        np.random.set_state(previous_rng)
        if target_device.type != "cuda":
            hand.data.qpos[:] = original_q
            mujoco.mj_kinematics(hand.model, hand.data)
            mujoco.mj_comPos(hand.model, hand.data)
            hand.jacobians = {}
    mesh = matching_mesh(hand.surface)
    rotations, translations = sample_original_poses(obj_loader, mesh, points, inplane, seed)
    count = len(translations)
    if not count:
        raise ValueError("Root pose sampler returned no proposals")

    def tensor(value):
        return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32, device=target_device)

    tmpl = {key: tensor(prepared[key])[None, None].expand(1, count, *np.shape(prepared[key])).contiguous()
            for key in ("grasp_qpos", "rot_c2h", "trans_c2h", "hand_cpn_c", "hand_skt_h", "n_evo")}
    obj = dict(rot_o2c_init=tensor(rotations)[None], trans_o2c_init=tensor(translations)[None],
               obj_scale=tensor([[1., 1., 1.]]), pose_w2so=tensor([[0., 0., 0., 1., 0., 0., 0.]]),
               ext_center=tensor([mesh.center_mass]), ext_wrench=tensor(np.zeros((1, 6))))
    environment = warp_util.WarpCollisionEnv(target_device)
    environment.load(None, [("template_mppi_object", mesh)])
    with torch.enable_grad():
        result = gen_init.HandObjMatcher(SimpleNamespace(opt_step=iterations), target_device, environment).forward(tmpl, obj)
    pose = result["grasp_pose"].detach().cpu().numpy().astype(np.float64)
    finite = np.isfinite(pose).all(axis=-1) & (np.linalg.norm(pose[:, 3:], axis=-1) > 1.e-12)
    indices = np.flatnonzero(finite)
    if not len(indices):
        raise RuntimeError("Root matcher produced no finite root candidates")
    pose = pose[indices]
    q = np.c_[pose[:, :3], Rotation.from_quat(pose[:, 3:], scalar_first=True).as_euler("XYZ"),
              np.broadcast_to(template.finger_qpos, (len(pose), 20))]
    losses = {key: result[key].detach().cpu().numpy()[indices].tolist() for key in ("loss_dist", "loss_normal")}
    correspondences = {key: result[key].detach().cpu().numpy()[indices].tolist()
                       for key in ("obj_cp_w", "obj_cn_w")}
    return q, dict(method="retained_HandObjMatcher_root_only", seed=seed, points=points, inplane=inplane,
                   iterations=iterations, sampled=count, finite_candidates=len(q),
                   nonfinite_candidates=count-len(q), proposal_indices=indices.tolist(), **losses,
                   **correspondences,
                   initialization_filter=False, qp_filter=False, closure_filter=False,
                   canonical_fk=fk_metadata,
                   finger_angles_unchanged=True, local_contacts_unchanged=True,
                   object_frame="fixed input mesh frame; no recentering or scaling")


def load_fixed_reference_template(reference_path, dexonomy_root, model, name):
    """Reuse the exact frozen slots after checking their original input hashes.

    A continuation must not re-run bone conversion or round mapped local points
    on a different machine. The reference template is a required input asset.
    """
    from .template_mppi_reference import template_signature
    reference_path = Path(reference_path)
    record = json.loads(reference_path.read_text())
    template = TemplateContacts(**json.loads(reference_path.with_name('template_contacts.json').read_text()))
    if record['template'] != name or record['template_signature'] != template_signature(template):
        raise ValueError('Frozen reference template signature mismatch')
    checkout = Path(dexonomy_root).resolve()
    provenance = template.provenance
    sources = {
        'annotation_sha256': checkout/'assets/hand/shadow/raw_anno'/f'{name}.yaml',
        'keypoints_sha256': checkout/'assets/hand/shadow/keypoint.yaml',
        'shadow_xml_sha256': checkout/'assets/hand/shadow/right.xml',
        'wuji_xml_sha256': asset_root('wuji2')/'right.xml',
    }
    if model_hand(model) != 'wuji2' or any(_sha(path) != provenance[key] for key,path in sources.items()):
        raise ValueError('Frozen template source assets changed')
    for body in template.body_names:
        model.body(body)  # Every fixed body must exist in the actual hand model.
        if _sha(query_path(model,body)) != provenance['query_sha256'][body]:
            raise ValueError(f'Frozen template query geometry changed: {body}')
    return template
