"""Original Dexonomy initialization with immutable, object-independent Wuji2 templates."""

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from .graspadmm_scene import required_body_groups, normalize_grasp_data
from .graspadmm_source import require_upstream


def _template_fingerprint(args, annotation_path):
    from retarget.config.run import ROOT
    from .hand import asset_root

    paths = {
        "wuji2_hand_xml": asset_root("wuji2") / "right.xml",
        "shadow_hand_xml": Path(args.dexonomy_root) / "assets/hand/shadow/right.xml",
        "annotation": Path(annotation_path),
        "bone_seed": ROOT / "retarget/env/grasp/seed.py",
        "bone_fk": ROOT / "retarget/env/grasp/template.py",
        "anatomical_joint_adapter": ROOT / "retarget/task/grasp/dexonomy.py",
        "anatomical_ik": ROOT / "retarget/initialization/ik.py",
        "hand_asset_adapter": ROOT / "retarget/env/grasp/hand.py",
        "conversion_cache_adapter": ROOT / "retarget/env/grasp/graspadmm_inputs.py",
    }
    fingerprints = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    identity = dict(version=1, source_sha256=fingerprints, torch=torch.__version__,
                    numpy=np.__version__, mujoco=mujoco.__version__,
                    gpu=torch.cuda.get_device_name(args.init_device))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, identity


def fixed_finger_angles(model, joints, args, annotation_path, *, cache=True, cache_root=None):
    """Cache only the deterministic, object-independent 20-joint conversion.

    The lock is shared by all workers in a frozen runtime. Matcher proposals,
    object contacts, ADMM states and successful grasps never enter this cache.
    """
    from retarget.config.run import ROOT
    from .seed import bone_seed_cuda

    begun = time.perf_counter()
    key, identity = _template_fingerprint(args, annotation_path)

    def convert():
        tick = time.perf_counter()
        qpos, info = bone_seed_cuda(model, joints, device=args.init_device)
        return qpos[0, 6:].copy(), info, time.perf_counter() - tick

    if not cache:
        qpos, info, seconds = convert()
        return qpos, dict(info, cache_hit=False, cache_enabled=False, template_cache_key=key,
                          template_conversion_seconds=seconds, template_cache_seconds=time.perf_counter() - begun)
    root = Path(cache_root) if cache_root is not None else ROOT / ".cache/graspadmm_templates"
    root.mkdir(parents=True, exist_ok=True)
    folder = root / key
    with (root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder / "finger_qpos.npy"
        metadata_path = folder / "identity.json"
        if path.is_file() and metadata_path.is_file():
            record = json.loads(metadata_path.read_text())
            qpos = np.load(path, allow_pickle=False)
            if (record.get("identity") == identity and qpos.shape == (20,)
                    and np.isfinite(qpos).all()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == record.get("qpos_sha256")):
                return qpos.copy(), dict(
                    method="cached_CUDA_batched_finger_bone_direction_LM", cache_hit=True, cache_enabled=True,
                    template_cache_key=key, template_cache_path=str(path),
                    template_conversion_seconds=0., template_cache_seconds=time.perf_counter() - begun,
                    original_conversion_seconds=record.get("conversion_seconds"),
                )
        qpos, info, seconds = convert()
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix="finger_qpos_", delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, qpos, allow_pickle=False)
        os.replace(temporary, path)
        record = dict(identity=identity, qpos_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      conversion_seconds=seconds, contents="Only 20 fixed finger joint angles; no object or solver results")
        with tempfile.NamedTemporaryFile(dir=folder, mode="w", prefix="identity_", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, sort_keys=True)
        os.replace(temporary, metadata_path)
        return qpos.copy(), dict(info, cache_hit=False, cache_enabled=True,
                                 template_cache_key=key, template_cache_path=str(path),
                                 template_conversion_seconds=seconds,
                                 template_cache_seconds=time.perf_counter() - begun)


def seed_trial(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def skeleton_arrays(hand):
    """Wuji counterparts of the upstream palm/proximal/middle line segments."""
    model, data = hand.model, hand.data
    hand.set(np.zeros(26))
    segments, bodies = [], []
    palm = model.body("right_palm_link").id
    palm_rotation = data.xmat[palm].reshape(3, 3)
    for finger in range(2, 6):
        child = model.body(f"right_finger{finger}_link1").id
        endpoint = (data.xpos[child] - data.xpos[palm]) @ palm_rotation
        segments.append(np.r_[[endpoint[0], endpoint[1], 0.], endpoint])
        bodies.append(palm)
    for finger in range(1, 6):
        for link in (2, 3):
            body = model.body(f"right_finger{finger}_link{link}").id
            child = model.body(f"right_finger{finger}_link{link + 1}").id
            endpoint = (data.xpos[child] - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            segments.append(np.r_[np.zeros(3), endpoint])
            bodies.append(body)
    return np.asarray(segments, np.float32), np.asarray(bodies)


def fixed_template(args, template_name, mesh_path, directory):
    """Map and pose once; no object contacts, refinement or acceptance fit."""
    from retarget.task.grasp.dexonomy import load_template, map_contacts, anatomical_joints
    from .inputs import build_scene, StaticHand, adapt_primitive_contacts

    model, surface, xml, offset = build_scene(mesh_path, args.scale, "wuji2")
    shadow, posed, contacts, annotation_path = load_template(args.dexonomy_root, template_name)
    contacts, contact_adaptations = adapt_primitive_contacts(args.dexonomy_root, template_name, contacts)
    # This mapping is immutable for a type, independent of the test object.
    bodies, local, normals, mapping = map_contacts(shadow, contacts, model)
    hand = StaticHand(model, surface, bodies, local, normals)
    fingers, bone_seed = fixed_finger_angles(model, anatomical_joints(shadow, posed)[None], args, annotation_path)
    q = np.r_[np.zeros(6), fingers]
    hand.set(q)
    world_points = hand.points(bodies, local)
    world_normals = np.einsum("nij,nj->ni", hand.data.xmat[bodies].reshape(-1, 3, 3), normals)
    body_names = [model.body(int(body)).name for body in bodies]
    required = required_body_groups(body_names)
    import yaml
    from retarget.task.grasp.dexonomy import wuji_body
    annotation = yaml.safe_load(annotation_path.read_text())
    if annotation.get("required_cbody") is not None:
        required = [list(dict.fromkeys(wuji_body(name) for name in group)) for group in annotation["required_cbody"]]
        if any(not group or set(group) - set(body_names) for group in required):
            raise ValueError("Explicit template required bodies must have annotated Wuji contacts")
    data = dict(
        tmpl_name=template_name, hand_cbody=body_names, required_cbody=required,
        grasp_qpos=np.r_[np.zeros(3), 1., np.zeros(3), q[6:]][None].astype(np.float32),
        hand_cpn_w=np.c_[world_points, world_normals].astype(np.float32),
        n_evo=np.array([0.], np.float32),
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    template_path = directory / "fixed_wuji2_template.npy"
    np.save(template_path, data)
    metadata = dict(
        source_annotation=str(annotation_path), mapping=mapping, bone_seed=bone_seed,
        primitive_contact_adaptations=contact_adaptations,
        policy="immutable_wuji2_template_no_object_fit_no_evolution", n_evo=0,
        required_cbody=required, object_recenter_offset=np.asarray(offset).tolist(),
        template_conversion_seconds=bone_seed["template_conversion_seconds"],
        template_cache_hit=bone_seed["cache_hit"],
    )
    (directory / "fixed_wuji2_template.json").write_text(json.dumps(metadata, indent=2))
    return hand, q, data, template_path, metadata


def initialize_one(hand, template_q, template, template_path, args, scene, *, seed, settings):
    """8192 native proposals, original filters/padding/FPS, one final candidate."""
    from .initialization import matching_mesh, sample_original_poses

    source = require_upstream()
    cfg = copy.deepcopy(source.init_config)
    if settings.get("init_epoch", 1) != 1 or settings.get("init_n_final", 1) != 1:
        raise ValueError("Independent single-candidate trials require init_epoch=1 and init_n_final=1")
    cfg.epoch = 1
    cfg.filter.general.n_final = 1
    cfg.matcher.opt_step = settings.get("init_iterations", 100)
    cfg.object.n_init_point = settings.get("init_points", 1024)
    cfg.object.n_init_inplane = settings.get("init_inplane", 8)
    seed_trial(seed)
    device = torch.device(args.init_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The original Dexonomy Warp initializer requires CUDA")
    torch.cuda.set_device(device)
    hand.set(template_q)
    loader = object.__new__(source.hand_loader.HandTemplateLoader)
    loader.skt_data, loader.skt_bid = skeleton_arrays(hand)

    def fk(fingers):
        hand.set(np.r_[np.zeros(6), fingers])
        return hand.data.xmat.reshape(-1, 3, 3).astype(np.float32), hand.data.xpos.astype(np.float32)

    prepared = loader._load_data(SimpleNamespace(forward_kinematics=fk), str(template_path))
    mesh = matching_mesh(hand.surface)
    rotations, translations = sample_original_poses(source.obj_loader, mesh, cfg.object.n_init_point, cfg.object.n_init_inplane, seed)
    count = len(translations)

    def tensor(value):
        return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32, device=device)

    tmpl_sample = {
        key: tensor(prepared[key])[None, None].expand(1, count, *np.shape(prepared[key])).contiguous()
        for key in ("grasp_qpos", "rot_c2h", "trans_c2h", "hand_cpn_c", "hand_skt_h", "n_evo")
    }
    # One immutable buffer entry: upstream's randint selection still consumes
    # its native RNG stream even though all indices refer to the same template.
    torch.randint(0, 1, (1, count))
    object_sample = dict(
        rot_o2c_init=tensor(rotations)[None], trans_o2c_init=tensor(translations)[None],
        obj_scale=tensor([[1., 1., 1.]]), pose_w2so=tensor([[0., 0., 0., 1., 0., 0., 0.]]),
        ext_center=tensor([mesh.center_mass]), ext_wrench=tensor(np.zeros((1, 6))),
    )
    collision = source.init.WarpCollisionEnv(device)
    collision.load(None, [(str(scene.mesh_path), mesh)])
    matcher = source.init.HandObjMatcher(cfg.matcher, device, collision)
    result = matcher.forward(tmpl_sample, object_sample)
    filter_instance = source.init.InitFilter(cfg.filter, device, collision)
    filter_counts = []
    # Each original filter and its original padding are invoked unchanged.
    for index, name in enumerate(filter_instance.valid_filter):
        before = len(result["obj_id"])
        result = filter_instance._wrapped_filter(result, name, len(filter_instance.valid_filter) - index - 1)
        filter_counts.append(dict(filter=name, before=before, after=len(result["obj_id"])))
    torch.cuda.synchronize(device)
    result = {key: value.detach().cpu().numpy() for key, value in result.items()}
    if len(result["grasp_pose"]) != 1:
        raise RuntimeError(f"Native n_final=1 produced {len(result['grasp_pose'])} candidates")
    # Exact sign convention of upstream operate_init's saved data.
    grasp = normalize_grasp_data(
        np.r_[result["grasp_pose"][0], result["grasp_qpos"][0]], scene,
        hand_cbody=template["hand_cbody"], hand_cpn_w=result["hand_cpn_w"][0],
        obj_cpn_w=np.c_[result["obj_cp_w"][0], -result["obj_cn_w"][0]],
        required_cbody=template["required_cbody"], template=template["tmpl_name"],
    )
    grasp["ext_center"] = result["ext_center"][0]
    grasp["ext_wrench"] = result["ext_wrench"][0]
    return grasp, dict(
        seed=seed, candidates=count, n_final=1, epochs=1, filter_counts=filter_counts,
        matcher_loss_dist=result["loss_dist"][0].tolist(), matcher_loss_normal=result["loss_normal"][0].tolist(),
        native_filter_padding=True, template_evolution=False,
        object_pose_identity=True, native_pose_filter_axis_branch="object_translation_norm_below_0.1m",
        config=source.OmegaConf.to_container(cfg, resolve=False),
    )
