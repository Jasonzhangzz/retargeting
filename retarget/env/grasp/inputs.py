"""Grasp geometry and the retained input initialization pipeline."""

import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from retarget.env.geometry import ObjectMeshSurface
from retarget.env.load_support import skew
from retarget.task.grasp.dexonomy import (
    ROOT,
    load_template,
    map_contacts,
    anatomical_joints,
    retarget_from_mano,
)


def build_scene(mesh, scale=1.0, hand_model="wuji1"):
    """Independent fixed object and 26-DOF Wuji, without custom query geoms."""
    from retarget.env.grasp.hand import hand_tree

    tree = hand_tree(hand_model)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == "geom":
                visual = child.get("group") == "1"
                child.set("contype", "0" if visual else "1")
                child.set("conaffinity", "0" if visual else "1")
    ET.SubElement(
        tree.find("asset"),
        "mesh",
        name="grasp_object_mesh",
        file=str(Path(mesh).resolve()),
        scale=" ".join([str(scale)] * 3),
    )
    body = ET.SubElement(tree.find("worldbody"), "body", name="grasp_object")
    geom = ET.SubElement(
        body,
        "geom",
        name="grasp_object_geom",
        mesh="grasp_object_mesh",
        type="mesh",
        contype="2",
        conaffinity="1",
        rgba=".25 .65 .85 1",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    surface = ObjectMeshSurface.from_model(model, model.body("grasp_object").id)
    vertices = surface.triangles.reshape(-1, 3)
    center = (vertices.min(0) + vertices.max(0)) / 2
    geom.set("pos", " ".join(map(str, -center)))
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    surface = ObjectMeshSurface.from_model(model, model.body("grasp_object").id)
    return model, surface, xml, center


class StaticHand:
    """Input-only Wuji kinematics and query assets; contains no grasp solver."""

    def __init__(self, model, surface, bodies, local, normals):
        from retarget.env.grasp.hand import HAND_BODIES, query_path, model_hand

        self.model, self.surface = model, surface
        self.hand_model = model_hand(model)
        self.data = mujoco.MjData(model)
        self.bodies, self.local, self.normals = bodies, local, normals
        self.all_bodies, self.all_local = [], []
        for name in HAND_BODIES:
            with np.load(query_path(model, name)) as asset:
                queries = asset["local_pos"]
            self.all_bodies.extend([model.body(name).id] * len(queries))
            self.all_local.extend(queries)
        self.all_bodies, self.all_local = (
            np.asarray(self.all_bodies),
            np.asarray(self.all_local),
        )

    def set(self, q):
        self.data.qpos[:26] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.jacobians = {}

    def points(self, bodies, local, jacobian=False):
        rotation = self.data.xmat[bodies].reshape(-1, 3, 3)
        offsets = np.einsum("nij,nj->ni", rotation, local)
        points = self.data.xpos[bodies] + offsets
        if not jacobian:
            return points
        for body in np.unique(bodies):
            if body not in self.jacobians:
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jp, jr, int(body))
                self.jacobians[body] = jp[:, :26], jr[:, :26]
        jp = np.array([self.jacobians[b][0] for b in bodies])
        jr = np.array([self.jacobians[b][1] for b in bodies])
        return points, jp - skew(offsets) @ jr, jr


def adapt_primitive_contacts(checkout, template, contacts):
    """Add the requested Power Sphere palm slot without editing vendor YAML."""
    if template.partition("_")[0] != "11" or any(body == "rh_palm" for body, _ in contacts):
        return contacts, []
    import yaml
    keypoint_path = Path(checkout) / "assets/hand/shadow/keypoint.yaml"
    keys = yaml.safe_load(keypoint_path.read_text())
    palm = np.asarray(keys["rh_palm"][1], dtype=float)
    return [*contacts, ("rh_palm", palm.copy())], [dict(
        body="rh_palm", adapted_local=palm.tolist(),
        source="local Power Sphere adaptation: Dexonomy central palm keypoint 1",
        reason="Power Sphere requires an explicit palm contact slot")]


def contactaware_proposal(hand,guess,args,primitive):
    """Keep pre-CA finite geometry if seed-mode CA throws or returns NaNs."""
    from .contactaware import refine_grasp_reference
    try:
        proposal=refine_grasp_reference(hand,guess,args,primitive=primitive)
        if getattr(args,'ca_as_seed',False) and any(not np.isfinite(value).all()
                for value in (proposal.hand_qpos,proposal.object_points,proposal.object_normals)):
            raise FloatingPointError('CA returned a nonfinite proposal')
        return proposal
    except Exception as error:
        if not getattr(args,'ca_as_seed',False):raise
        guess.metadata.update(initialization_valid=False,contactaware=dict(
            success=False,error_type=type(error).__name__,error=str(error),
            fallback='finite pre-CA geometric alignment; initialization only'))
        return guess


def initialize_cases(args, output):
    contactaware_mode = getattr(args, "reference_method", "template") == "dexforge_contactaware"
    from retarget.env.grasp.primitives import jobs, catalog
    from retarget.env.grasp.initialization import (
        global_alignment,
        pose_in_hand,
        validate_initialization,
    )
    from retarget.env.grasp.template import fit_contact_template
    from retarget.env.grasp.input_geometry import (
        dense_power_template,
        fit_power_cylinder_input,
    )
    from scipy.spatial.transform import Rotation
    import warp as wp

    if hasattr(args, "_tasks"):
        tasks = args._tasks
    elif args.primitives:
        tasks, skipped = jobs(output, args.templates, args)
        (output / "primitive_catalog.json").write_text(
            json.dumps(catalog(args.dexonomy_root), indent=2)
        )
        (output / "skipped.json").write_text(json.dumps(skipped, indent=2))
    else:
        tasks = [
            (p.expanduser().resolve(), name, None)
            for p in args.object
            for name in args.templates
        ]
    from retarget.task.grasp.wrist_prior import (
        initialize_references,
        configure_reference,
        save_references,
    )

    initialize_references(args, tasks, output)
    rng = np.random.default_rng(args.seed)
    if hasattr(args, "_rng_state"):
        rng.bit_generator.state = args._rng_state
    summary = []
    for object_index, (mesh, name, primitive) in enumerate(tasks):
        import copy

        object_rng_state = copy.deepcopy(rng.bit_generator.state)
        begun = time.perf_counter()
        model, surface, input_xml, offset = build_scene(
            mesh, args.scale, getattr(args, "hand_model", "wuji1")
        )
        shadow, posed, contacts, source = load_template(args.dexonomy_root, name)
        from retarget.env.grasp.tip_patch import (
            centered_template_contacts,
            configure_tip_centers,
        )

        contacts, primitive_adaptations = adapt_primitive_contacts(
            args.dexonomy_root, name, contacts)
        contacts, contact_adaptations = centered_template_contacts(
            args.dexonomy_root, name, contacts
        )
        contact_adaptations = primitive_adaptations + contact_adaptations
        semantic_patches = None
        if getattr(args, "contact_mapping", "nearest") == "semantic_patch":
            from retarget.env.grasp.semantic_patches import match_semantic_patches

            bodies, local, normals, mapping, semantic_patches = match_semantic_patches(
                shadow, contacts, model
            )
        else:
            bodies, local, normals, mapping = map_contacts(shadow, contacts, model)
        hand = StaticHand(model, surface, bodies, local, normals)
        configure_tip_centers(hand, name, mapping,
            semantic_masks=None if semantic_patches is None else [patch["mask"] for patch in semantic_patches])
        semantic_reference_records = None
        if semantic_patches is not None:
            semantic_reference_records = copy.deepcopy(mapping)
            hand.semantic_contact_masks = [patch["mask"].copy() for patch in semantic_patches]
            hand.semantic_patch_labels = [patch["id"] for patch in semantic_patches]
            # Fixed matched-template references precede all pose fitting and
            # object-dependent query updates. Their patch identities never move.
            hand.semantic_contact_local = hand.local.copy()
            hand.semantic_contact_normals = hand.normals.copy()
            for i, body in enumerate(hand.bodies):
                tip = getattr(hand, "tip_patches", {}).get(int(body))
                if tip is not None:
                    hand.semantic_contact_masks[i] &= tip["mask"]
                    if not hand.semantic_contact_masks[i][tip["index"]]:
                        raise ValueError("Central tip policy is outside its matched semantic patch")
        from retarget.env.grasp.seed import bone_seed_cuda

        fitted, bone_seed = bone_seed_cuda(
            model, anatomical_joints(shadow, posed)[None], device=args.init_device
        )
        reference_qpos, reference_normals = fitted[0].copy(), normals.copy()
        hand.ik_reference_qpos = reference_qpos
        hand.contact_reference_normals = reference_normals
        from retarget.env.grasp.orientation import configure_orientation

        configure_orientation(hand, name, primitive)
        configure_reference(hand, name, primitive, args)
        from retarget.env.grasp.intent import configure_intent
        configure_intent(hand, name, primitive, args)
        hand.primitive = primitive
        print(f"[grasp] {mesh.name} / {name}", flush=True)
        template_fit = dict(method="bone_direction_IK_only", enabled=False)
        initialization_seconds = 0.0
        if not getattr(args, "no_template_fit", False):
            opening = (
                2 * primitive["radius_m"]
                if primitive and primitive["kind"] in ("cylinder", "sphere")
                else None
            )
            fitted_q, template_fit = fit_contact_template(
                hand,
                shadow,
                posed,
                contacts,
                fitted[0],
                device=args.init_device,
                opening=opening,
                joint_prior=1.0 if getattr(args, "load_conditioning", False) else None,
                reference_qpos=reference_qpos,
                reference_normals=reference_normals,
            )
            fitted[0] = fitted_q
            final_fit = template_fit
            if contactaware_mode:
                for i, record in enumerate(mapping):
                    record["initial_query_index"] = record["query_index"]
                    record.update(
                        query_index=template_fit["surface_query_indices"][i],
                        wuji_local=hand.local[i].tolist(),
                        wuji_local_normal=hand.normals[i].tolist(),
                    )
        from .orientation import primitive_pose_seeds
        axis_seeds = primitive_pose_seeds(hand, fitted[0])
        guesses = global_alignment(hand, fitted[0], args, rng,
            **({"pose_seed": axis_seeds[0]} if len(axis_seeds) else {}))
        initialization_seconds += guesses[0].metadata.get("seconds", 0.0)
        if contactaware_mode:
            # Template pose and original alignment are seeds only. ContactAware
            # computes the reference from the actual primitive and accepts it
            # by contact/geometry accuracy, independently of shape-first flags.
            guesses=[contactaware_proposal(hand,guess,args,primitive) for guess in guesses]
            template_fit = dict(
                method="DexForge_ContactAware_joint_contact_and_IK",
                enabled=True,
                target_source="ContactAware optimized primitive anchors and hand IK",
                template_joint_angles_are_reference=False,
                numerical_seed_fit=template_fit,
            )
        else:
            for guess in guesses:
                validate_initialization(hand, guess, args, reference_normals)
        if not contactaware_mode and not getattr(args, "no_template_fit", False):
            # The first original initialization supplies attainable object
            # surface targets. Fit the Wuji hand to those contacts, then run
            # the original initializer again so all final poses/contacts remain
            # its unmodified outputs. No extra object-pose optimizer.
            def quality(guess):
                info = guess.metadata
                return (
                    not info.get("initialization_valid", False),
                    max(1.0 - info.get("forbidden_region_min_gap_mm", 1.0), 0.0),
                    max(
                        max(
                            info.get("initial_penetration_mm", float("inf")),
                            info.get("initial_self_penetration_mm", 0.0),
                        )
                        - args.penetration_tolerance_mm,
                        0.0,
                    ),
                    info.get("bone_direction_max_deviation_deg", 0.0),
                    info.get(
                        "selection_objective", info.get("objective", float("inf"))
                    ),
                )

            template_fit["object_contact_refinements"] = []
            for refinement_index in range(getattr(args, "template_refinements", 2)):
                guide = min(guesses, key=quality)
                if guide.metadata.get("initialization_valid", False):
                    break
                saved_local, saved_normals = hand.local.copy(), hand.normals.copy()
                reserve_weight = None
                if getattr(args, "load_conditioning", False):
                    reserve_weight = (
                        1.0
                        if (guide.metadata.get("load_conditioning") or {}).get(
                            "feasible"
                        )
                        else 2.0 ** (refinement_index + 1)
                    )
                power_cylinder = dense_power_template(hand, primitive)
                if power_cylinder:
                    refined_q, refinement = fit_power_cylinder_input(
                        hand,
                        guide.hand_qpos,
                        device=args.init_device,
                        joint_prior=reserve_weight,
                        reference_qpos=reference_qpos,
                        reference_normals=reference_normals,
                    )
                else:
                    refined_q, refinement = fit_contact_template(
                        hand,
                        shadow,
                        posed,
                        contacts,
                        guide.hand_qpos,
                        device=args.init_device,
                        object_contacts=(guide.object_points, -guide.object_normals),
                        joint_prior=reserve_weight,
                        reference_qpos=reference_qpos,
                        reference_normals=reference_normals,
                    )
                refined_guesses = global_alignment(
                    hand, refined_q, args, rng, pose_seed=refined_q
                )
                for guess in refined_guesses:
                    validate_initialization(hand, guess, args, reference_normals)
                initialization_seconds += refined_guesses[0].metadata.get(
                    "seconds", 0.0
                )
                accepted = min(map(quality, refined_guesses)) <= min(
                    map(quality, guesses)
                )
                template_fit["seconds"] += refinement["seconds"]
                template_fit["object_contact_refinements"].append(
                    dict(
                        refinement,
                        accepted=accepted,
                        source_initialization=guide.metadata.copy(),
                        refined_initialization=min(
                            refined_guesses, key=quality
                        ).metadata.copy(),
                        source_object_points=guide.object_points.tolist(),
                        source_object_inward_normals=(-guide.object_normals).tolist(),
                    )
                )
                if accepted:
                    guesses = refined_guesses
                    fitted[0], final_fit = refined_q, refinement
                    template_fit["after"] = refinement["after"]
                else:
                    hand.local, hand.normals = saved_local, saved_normals
                    if getattr(args, "load_conditioning", False) and not (
                        guide.metadata.get("load_conditioning") or {}
                    ).get("feasible"):
                        continue
                    break
                best_input = min(guesses, key=quality).metadata
                if (
                    best_input.get("initialization_valid", False)
                    and max(
                        best_input.get("matching", {}).get("gap_mm", [float("inf")])
                    )
                    <= args.contact_tolerance_mm
                ):
                    break
            local, normals = hand.local, hand.normals
            for i, record in enumerate(mapping):
                record["initial_query_index"] = record["query_index"]
                record["initial_mapping_distance_mm"] = record.pop(
                    "mapping_distance_mm"
                )
                record.update(
                    query_index=final_fit["surface_query_indices"][i],
                    wuji_local=local[i].tolist(),
                    wuji_local_normal=normals[i].tolist(),
                    template_position_error_mm=final_fit["contact_position_error_mm"][
                        i
                    ],
                    template_normal_error_deg=final_fit["contact_normal_error_deg"][i],
                )
        for guess in guesses:
            guess.metadata["template_fit"] = template_fit
            guess.metadata["bone_seed"] = bone_seed
            if semantic_patches is not None:
                guess.metadata["contact_mapping"] = "semantic_patch"
                guess.metadata["semantic_patch_labels"] = hand.semantic_patch_labels
            if contact_adaptations:
                guess.metadata["contact_adaptations"] = contact_adaptations
        case = output / f"{object_index:03d}_{mesh.stem}" / name
        case.mkdir(parents=True)
        if contactaware_mode:
            from retarget.common.results import write_json

            write_json(case / "contactaware.json", [guess.metadata["contactaware"] for guess in guesses])
            accepted = [i for i, guess in enumerate(guesses) if guess.metadata["initialization_valid"]]
            np.savez_compressed(
                case / "contactaware_reference.npz",
                accepted_seed_indices=accepted,
                accepted_reference_qpos=np.asarray([
                    guesses[i].contactaware_reference_qpos for i in accepted
                ]).reshape(-1, 26),
                proposed_hand_qpos=np.stack([guess.hand_qpos for guess in guesses]),
                proposed_object_anchors=np.stack([guess.object_points for guess in guesses]),
            )
        if semantic_patches is not None:
            (case / "semantic_matching.json").write_text(
                json.dumps(dict(
                    method="intrinsic_semantic_surface_patch_matching_before_gesture_IK",
                    records=semantic_reference_records,
                    fitted_contacts=[dict(
                        patch=hand.semantic_patch_labels[i],
                        query_index=record["query_index"],
                        local=hand.local[i].tolist(),
                        normal=hand.normals[i].tolist(),
                    ) for i, record in enumerate(mapping)],
                    patch_labels=hand.semantic_patch_labels,
                    reference_local=hand.semantic_contact_local.tolist(),
                    reference_normals=hand.semantic_contact_normals.tolist(),
                    candidate_counts=[int(mask.sum()) for mask in hand.semantic_contact_masks],
                ), indent=2)
            )
            np.savez_compressed(
                case / "semantic_patches.npz",
                **{f"mask_{i:03d}": mask for i, mask in enumerate(hand.semantic_contact_masks)},
            )
        yield hand, guesses, case, input_xml
