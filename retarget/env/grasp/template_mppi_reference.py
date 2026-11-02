"""Fit an object in a fixed template wrist at 1x, then share that pose.

The retained GraspADMM/Dexonomy matcher optimizes the inverse transform (the
hand in the object frame). Inverting it gives the requested object-in-wrist
pose. Keeping object coordinates as world coordinates lets the existing MPPI
and replay tools use the same geometry without changing their frame convention.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.common.results import write_json


POLICY = "template_1x_object_in_wrist_v1"
GRASP_POLICY = "template_1x_solved_grasp_reference_v2"


def template_signature(template):
    value = dict(body_names=list(template.body_names),
                 local_points=np.asarray(template.local_points).tolist(),
                 local_normals=np.asarray(template.local_normals).tolist(),
                 finger_qpos=np.asarray(template.finger_qpos).tolist())
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pose_matrix(q):
    q = np.asarray(q, float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("XYZ", q[3:6]).as_matrix()
    matrix[:3, 3] = q[:3]
    return matrix


def select_match(candidates, report):
    """Use exactly the retained matcher's all-slot distance/normal objective."""
    q = np.asarray(candidates, float)
    if q.ndim != 2 or q.shape[1] != 26 or not len(q):
        raise ValueError("Canonical matcher must return nonempty [N,26] poses")
    distance = np.asarray(report["loss_dist"], float)
    normal = np.asarray(report["loss_normal"], float)
    if (distance.ndim != 2 or distance.shape != normal.shape
            or distance.shape[0] != len(q) or not distance.shape[1]):
        raise ValueError("Canonical matcher losses must have shape [candidates, contacts]")
    cost = (1000. * distance + normal).mean(axis=-1)
    valid = np.isfinite(q).all(axis=-1) & np.isfinite(cost)
    if not valid.any():
        raise FloatingPointError("No finite canonical object pose match")
    index = int(np.argmin(np.where(valid, cost, np.inf)))
    return q[index].copy(), dict(selected=index, candidate_costs=cost.tolist(),
        selection_objective="retained_matcher_mean_1000_distance_squared_plus_normal_squared",
        root_selection_cost="canonical_1x_template_distance_and_normal",
        actual_contact_or_force_closure_required=False)


def reference_key(name, primitive):
    return name, primitive["kind"] if primitive else "custom"


def fit_reference(hand, template, *, name, directory, primitive, seed, device, initialization,
                  canonical_contacts=None, selection_settings=None, object_mass=.1):
    from .initialization import matching_mesh
    from .template_mppi_inputs import initialize_root_candidates

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    initialization = dict(initialization)
    mode = initialization.pop("selection", "matcher")
    if mode not in ("matcher", "collision_aware"):
        raise ValueError("Unknown canonical candidate selection")
    candidates, report = initialize_root_candidates(hand, template, seed=seed, device=device,
                                                     canonical_contacts=canonical_contacts, **initialization)
    initial, selection = select_match(candidates, report)
    mesh = directory / "object_1x.obj"
    matching_mesh(hand.surface).export(mesh, digits=17)
    if mode == "collision_aware":
        from .template_mppi import TemplatePoseObjective
        import torch
        objective = TemplatePoseObjective(mesh, selection_settings, device=device, object_mass=object_mass)
        objective.bind(template, initial)
        if hasattr(objective, 'configure_object_regions'):
            objective.configure_object_regions(name, primitive)
        if hasattr(objective, 'posture'):
            # This stage fits only the root with the original mapped angles.
            # Grasp articulation ranges apply after the subsequent size IK,
            # which is the first stage permitted to change finger angles.
            objective.posture = None
            objective.low = objective.physical_low.clone()
            objective.high = objective.physical_high.clone()
        objective.set_batch_size(256)
        with torch.no_grad():
            costs = objective.evaluate_initialization(torch.as_tensor(candidates, device=device,
                dtype=torch.float32)[:, None]).cpu().numpy()
        finite = np.isfinite(costs)
        if not finite.any():
            raise FloatingPointError("No finite canonical collision-aware candidate")
        index = int(np.argmin(np.where(finite, costs, np.inf)))
        initial = candidates[index].copy()
        selection.update(matcher_selected=selection['selected'], matcher_candidate_costs=selection['candidate_costs'],
            selected=index, candidate_costs=costs.tolist(),
            selection_objective="template_initialization_geometry_cost_including_full_hand_penetration",
            root_selection_cost="canonical_1x_template_distance_normal_and_full_hand_penetration",
            candidate_selection="collision_aware", closure_solver_used_for_selection=False)
        del objective
    if not np.array_equal(candidates[:, 6:], np.broadcast_to(template.finger_qpos, (len(candidates), 20))):
        raise ValueError("Canonical object fit changed the template finger angles")
    object_in_wrist = np.linalg.inv(pose_matrix(initial))
    projections = np.asarray(report.pop("obj_cp_w"), float)
    inward_normals = np.asarray(report.pop("obj_cn_w"), float)
    expected = (len(candidates), len(template.body_names), 3)
    if (projections.shape != expected or inward_normals.shape != expected
            or not np.isfinite(projections[selection["selected"]]).all()
            or not np.isfinite(inward_normals[selection["selected"]]).all()):
        raise ValueError("Invalid canonical object contact correspondences")
    selected_points = projections[selection["selected"]]
    # Retained WarpCollisionEnv returns inward object normals for hand alignment.
    selected_outward = -inward_normals[selection["selected"]]
    write_json(directory / "template_contacts.json", template.to_dict())
    np.savez_compressed(directory / "candidates.npz", qpos=candidates,
                        object_points=projections, object_inward_normals=inward_normals)
    record = dict(policy=POLICY, template=name, canonical_primitive=primitive,
        canonical_size_scale=1., template_signature=template_signature(template),
        canonical_mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        initial_qpos=initial.tolist(), object_in_wrist=object_in_wrist.tolist(),
        wrist_in_object=pose_matrix(initial).tolist(), wrist_body="grasp_wrist_rz",
        transform_convention="T_A_B maps coordinates in B into A; object world pose is identity",
        pose_reuse="same rotation and translation at every size; translation is never scaled",
        object_points=selected_points.tolist(), object_outward_normals=selected_outward.tolist(),
        object_points_in_wrist=(selected_points @ object_in_wrist[:3, :3].T + object_in_wrist[:3, 3]).tolist(),
        object_outward_normals_in_wrist=(selected_outward @ object_in_wrist[:3, :3].T).tolist(),
        matcher=report, **selection, seconds=time.perf_counter() - started,
        template_finger_angles_preserved=True, template_contacts_preserved=True, opening_prefit=False)
    path = directory / "reference.json"
    write_json(path, record)
    return path.resolve()


def load_reference(path, template, name):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if (record.get("policy") not in (POLICY, GRASP_POLICY, "template_adjacent_solved_grasp_v1") or record.get("template") != name
            or record.get("template_signature") != template_signature(template)):
        raise ValueError("Canonical object reference does not match the fixed template")
    q = np.asarray(record["initial_qpos"], float)
    solved = record.get('object_contact_reference') in ('solved_1x_grasp', 'solved_adjacent_grasp')
    template_q = np.asarray(record.get('root_match_initial_qpos', q) if solved else q, float)
    if q.shape != (26,) or not np.isfinite(q).all() or not np.array_equal(template_q[6:], template.finger_qpos):
        raise ValueError("Invalid canonical reference hand pose")
    if not np.allclose(pose_matrix(q) @ np.asarray(record["object_in_wrist"]), np.eye(4), atol=1.e-10, rtol=0):
        raise ValueError("Inconsistent canonical object/wrist transforms")
    return q.copy(), dict(policy=record['policy'], reference=str(path),
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_size_scale=record.get("canonical_size_scale", 1.), canonical_primitive=record["canonical_primitive"],
        object_in_wrist=record["object_in_wrist"], wrist_in_object=record["wrist_in_object"],
        pose_reused_without_resampling=True, translation_scaled=False,
        canonical_fit_seconds=record["seconds"], selected=record["selected"],
        canonical_fk=record["matcher"].get("canonical_fk"),
        candidate_costs=record["candidate_costs"], selection_objective=record["selection_objective"],
        template_finger_angles_preserved=bool(np.array_equal(q[6:], template.finger_qpos)),
        template_contacts_preserved=True, opening_prefit=False,
        object_contact_reference=record.get('object_contact_reference', 'root_matcher'),
        canonical_grasp_reference=record.get('canonical_grasp_reference'),
        canonical_grasp_seconds=record.get('canonical_grasp_seconds', 0.))


def prepare_reference(mesh, name, primitive, args, output, *, device, seed, template=None,
                      canonical_contacts=None):
    """Build an exact 1x scene even when that size is absent from the job grid."""
    from .inputs import build_scene, StaticHand
    from .primitives import canonical_specification, primitive_mesh
    from .template_mppi_inputs import load_solver_template_contacts

    family = reference_key(name, primitive)[1]
    directory = Path(output) / "template_initialization" / f"{name}_{family}"
    canonical = None
    if primitive:
        canonical = canonical_specification(int(name.partition("_")[0]), family, args)
        mesh = primitive_mesh(directory / "source_1x.obj", family,
                              canonical["radius_m"], canonical["height_m"])
    model, surface, xml, offset = build_scene(mesh, args.scale, "wuji2")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scene_1x.xml").write_text(xml)
    if template is None:
        template = load_solver_template_contacts(name, args.dexonomy_root, model, device=device,
                                          cache_root=Path(output) / "template_cache")
    bodies = np.array([model.body(body).id for body in template.body_names])
    hand = StaticHand(model, surface, bodies, template.local_points.copy(), template.local_normals.copy())
    write_json(directory / "object_frame.json", dict(source_mesh=str(Path(mesh).resolve()),
        scale=args.scale, recenter_offset_m=np.asarray(offset).tolist()))
    print(f"[template_mppi] {name}/{family}: fitting object pose at 1x", flush=True)
    path = fit_reference(hand, template, name=name, directory=directory, primitive=canonical,
                         seed=seed, device=device, initialization=args.settings.initialization,
                         canonical_contacts=canonical_contacts, selection_settings=args.settings.objective,
                         object_mass=args.settings.object_mass)
    if args.settings.size_ik.enabled and args.settings.size_ik.bootstrap_canonical:
        attach_canonical_grasp(path, template, args, name, device, seed)
    return path


def attach_canonical_grasp(path, template, args, name, device, seed):
    """Use a solved 1x grasp as contact/angle source, not root-matcher proposals."""
    from .template_mppi import TemplatePoseObjective
    from dataclasses import replace
    if not hasattr(TemplatePoseObjective, 'initialize_size'):
        return  # Lightweight test doubles implement the legacy runner contract.
    path = Path(path)
    record = json.loads(path.read_text())
    canonical = record['canonical_primitive']
    directory = args.settings.size_ik.canonical_grasp_directory
    started = time.perf_counter()
    superseded_cache = None
    if directory is not None and template.provenance.get('policy') == 'fixed_prismatic_pad_center_v1':
        source = Path(directory).expanduser().resolve()/f'{name}.json'
        cached = json.loads(source.read_text())
        if cached['template_signature'] != template_signature(template):
            # The user's pad-center policy changes the contact template. Old
            # tip-based 1x grasps are evidence only, never initialization seeds.
            superseded_cache = dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reason='new fixed pad-center contact template requires a fresh canonical solve')
            directory = None
    if directory is not None:
        if canonical is None:
            raise ValueError('A cached canonical grasp requires a registered primitive')
        source = Path(directory).expanduser().resolve()/f'{name}.json'
        cached = json.loads(source.read_text())
        if cached['template_signature'] != template_signature(template):
            raise ValueError(f'Canonical grasp template mismatch: {name}')
        for key in ('kind', 'radius_m', 'height_m'):
            if cached['primitive'][key] != canonical[key]:
                raise ValueError(f'Canonical grasp dimensions mismatch: {name}, {key}')
        for source_file, digest in cached['source_sha256'].items():
            if hashlib.sha256(Path(source_file).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Canonical grasp source changed: {source_file}')
        q = np.asarray(cached['qpos'], float)
        points = np.asarray(cached['object_points'], float)
        normals = np.asarray(cached['object_outward_normals'], float)
        evidence = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        policy='audited_solved_1x_grasp_contacts', from_cache=True)
    else:
        if name.startswith('9_') and canonical and canonical['kind'] == 'cylinder':
            from .template_mppi_pinch import bootstrap_palmar_pinch
            q,points,normals=bootstrap_palmar_pinch(path,template,args,device,seed)
        else:
            q,points,normals=_fresh_canonical_grasp(path,template,args,name,canonical,device,seed)
        evidence = dict(source=str(path.with_name('canonical_grasp.json')),
            sha256=hashlib.sha256(path.with_name('canonical_grasp.json').read_bytes()).hexdigest(),
            policy='fresh_1x_ik_and_mppi_grasp_contacts', from_cache=False)
        if superseded_cache is not None:
            evidence['superseded_cache'] = superseded_cache
    if (q.shape != (26,) or points.shape != (len(template.body_names), 3) or normals.shape != points.shape
            or not np.isfinite(q).all() or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(normals, axis=-1), 1., atol=2.e-5)):
        raise ValueError(f'Invalid solved canonical grasp: {name}')
    canonical_side_slots = []
    if name.startswith(('6_', '7_', '9_', '10_')) and canonical and canonical['kind'] in ('cylinder','disk'):
        import torch
        projector = TemplatePoseObjective(path.with_name('object_1x.obj'),args.settings.objective,
            device=device,object_mass=args.settings.object_mass)
        projector.bind(template,q)
        projector.configure_object_regions(name,canonical)
        projector.set_batch_size(1)
        with torch.no_grad():
            geometry = projector._geometry(torch.as_tensor(q,device=device,dtype=torch.float32)[None])
        points, normals = geometry[2][0].cpu().numpy(),geometry[3][0].cpu().numpy()
        canonical_side_slots = projector.side_projection.slots.cpu().tolist() if projector.side_projection is not None else []
        del projector
    record.update(policy=GRASP_POLICY, root_match_initial_qpos=record['initial_qpos'],
        root_match_object_in_wrist=record['object_in_wrist'],
        root_match_wrist_in_object=record['wrist_in_object'],
        initial_qpos=q.tolist(), wrist_in_object=pose_matrix(q).tolist(),
        object_in_wrist=np.linalg.inv(pose_matrix(q)).tolist(),
        matcher_object_points=record['object_points'],
        matcher_object_outward_normals=record['object_outward_normals'],
        canonical_grasp_qpos=q.tolist(), object_points=points.tolist(),
        object_outward_normals=normals.tolist(), canonical_grasp_reference=evidence,
        canonical_side_contact_slots=canonical_side_slots,
        canonical_correspondences_reprojected=bool(canonical_side_slots),
        object_contact_reference='solved_1x_grasp', canonical_grasp_seconds=time.perf_counter()-started)
    transform = np.array(record['object_in_wrist'])
    record['object_points_in_wrist'] = (points@transform[:3, :3].T+transform[:3, 3]).tolist()
    record['object_outward_normals_in_wrist'] = (normals@transform[:3, :3].T).tolist()
    write_json(path, record)


def _fresh_canonical_grasp(path, template, args, name, canonical, device, seed):
    from .template_mppi import TemplatePoseObjective
    from dataclasses import replace

    objective = TemplatePoseObjective(path.with_name('object_1x.obj'), args.settings.objective,
                                     device=device, object_mass=args.settings.object_mass)
    q, _ = load_reference(path, template, name)
    objective.bind(template, q)
    objective.configure_object_regions(name, canonical)
    ik_settings = replace(args.settings.size_ik, bootstrap_canonical=False)
    q, ik = objective.initialize_size(path, canonical, q, ik_settings, seed)
    q, optimization = objective.optimize(q, args.solver, seed)
    diagnostics = objective.final_diagnostics(q)
    points = np.asarray(diagnostics['projected_points'], float)
    normals = np.asarray(diagnostics['object_normals'], float)
    write_json(path.with_name('canonical_grasp.json'), dict(qpos=q, diagnostics=diagnostics,
        ik=ik, optimization=optimization))
    del objective
    return q, points, normals


def prepare_canonical_templates(args, names, output, *, device):
    import mujoco
    import xml.etree.ElementTree as ET
    from .hand import hand_tree
    from .template_mppi_inputs import load_solver_template_contacts, canonical_template_contacts

    model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))
    templates = {name: load_solver_template_contacts(name, args.dexonomy_root, model, device=device,
                                             cache_root=Path(output) / "template_cache")
                 for name in dict.fromkeys(names)}
    contacts = canonical_template_contacts(model, list(templates.values()), device=device)
    return templates, dict(zip(templates, contacts))


def prepare_all_references(config, output, tasks):
    """Spawn entry: release the initialization CUDA context before MPPI workers."""
    import torch
    import warp as wp
    from .template_mppi_runner import validate

    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"
    wp.init()
    wp.set_device(config.device)
    args = validate(config)
    templates, canonical = prepare_canonical_templates(args, [name for _, name, _ in tasks],
                                                       output, device=config.device)
    references = {}
    paths = []
    for mesh, name, primitive in tasks:
        key = reference_key(name, primitive)
        if key not in references:
            references[key] = str(prepare_reference(mesh, name, primitive, args, output,
                device=config.device, seed=config.seed, template=templates.get(name),
                canonical_contacts=canonical.get(name)))
        paths.append(references[key])
    write_json(Path(output) / "template_initialization" / "manifest.json", dict(
        policy=GRASP_POLICY if args.settings.size_ik.enabled and args.settings.size_ik.bootstrap_canonical else POLICY,
        fits=len(references), cases=len(tasks), references=paths,
        fk_backend="torch_cuda_batched_WujiContactKinematics", fk_batch_size=len(templates)))
    return paths
