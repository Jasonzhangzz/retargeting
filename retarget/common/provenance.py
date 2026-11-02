"""Versions and hashes for the actual local inputs and implementation."""

import hashlib
import importlib.metadata
from pathlib import Path
import torch
from retarget.config.run import ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(config, cases, output):
    versions = {}
    for name in (
        "torch",
        "warp-lang",
        "mujoco",
        "mujoco-warp",
        "numpy",
        "scipy",
        "viser",
        "cyipopt",
        "clarabel",
        "qpsolvers",
        "trimesh",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    paths = [config.source]
    data = Path(config.task["data"])
    paths.extend(data.glob("*.yaml"))
    paths.extend(data.glob("*.npz"))
    paths.extend(data.glob("*.xml"))
    for case in cases:
        model = Path(case["model"])
        model = model if model.is_absolute() else output / model
        paths.append(model)
        if config.task.get("type") == "grasp":
            paths.extend(model.parent.glob("input_*/card_input.npz"))
            paths.extend(model.parent.glob("input_*/contactaware_reference.json"))
            case_root = (output / case["trajectory"]).parent
            paths.extend(case_root / name for name in ("card_input.npz", "contactaware_reference.json",
                "reference_manifest.json", "forceaware_model.yaml", "native_scene/scene_manifest.json",
                "control_preload.npz", "control_preload.json", "control_initialization.json",
                "initial_state_policy.json", "dynamics_adapter.json", "ca_input/ca_refinement_manifest.json"))
    result = dict(
        versions=versions,
        gpu=torch.cuda.get_device_name(config.device) if str(config.device).startswith("cuda") and torch.cuda.is_available() else None,
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "retarget").rglob("*.py")
        },
        input_sha256={str(p): sha256(p) for p in paths if p.is_file()},
        cost_environment_scope="Installed local engines; source-equivalent costs do not imply bitwise upstream dynamics",
    )
    if config.solver.type == "graspadmm":
        import json
        manifest = ROOT / "third_party/Dexonomy/GRASPADMM_UPSTREAM.json"
        result["graspadmm"] = json.loads(manifest.read_text())
        result["graspadmm_source_manifest_sha256"] = sha256(manifest)
        result["cost_environment_scope"] = (
            "Pinned original GraspADMM ADMM/QP/filter/pregrasp/eval implementation; "
            "WUJI2 geometry, joint/frame, rigid pad ownership and input template adapters")
    if config.task.get("type") == "grasp":
        result["evaluation_protocol"] = "graspadmm_native_v1"
        result["evaluation_runtime"] = str(ROOT/".venv-graspadmm/bin/python")
    if config.solver.type == "dexforge":
        from retarget.dynamics.dexforge_upstream import verify_upstream

        result["dexforge"] = verify_upstream()
        result["cost_environment_scope"] = (
            "Pinned upstream DexForge original ForceAware workflow, objective, native planning/execution and mesh replay; WUJI2 assets/topology/CA-input adapters"
            if config.task.get("type") == "grasp" else
            "Pinned upstream DexForge native ForceAware planning and execution; G1 objective/scene adapters"
        )
        if config.task.get("type") == "grasp" and config.environment.get("dexforge", {}).get("servo_integration") == "implicit_actuator":
            result["cost_environment_scope"] = (
                "Pinned upstream DexForge FA objective/optimizer/workflow with explicit local clamp-aware actuator damping integration; "
                "vendor source bytes are verified but the native runtime integration is extended"
            )
            result["native_integration_adapter"] = "retarget/dynamics/dexforge_servo_integration.py"
        if config.task.get("type") == "grasp":
            options = config.environment.get("dexforge", {})
            result["grasp_initialization"] = dict(control_preload=options.get("control_preload", False),
                fixed_initial_state=options.get("fixed_initial_state", False))
    if config.task.get("type") == "grasp" and config.environment.get("spider") is not None:
        from retarget.env.grasp.spider_source import verify_sources
        result["spider"] = verify_sources()
        result["contactaware_inputs_changed"] = False
        result["forceaware_optimizer_used"] = False
        result["cost_environment_scope"] = "SPIDER dex MPPI/tracking semantics with accepted unchanged CA references and native MJWarp mesh dynamics"
        cost = config.environment["spider"].get("cost", {})
        keep_scale = cost.get("contact_keep_rew_scale", 0.)
        result["local_cost_extensions"] = dict(ca_finger_contact_keep=dict(
            enabled=keep_scale > 0, weight=keep_scale,
            minimum_normal_force_n=cost.get("contact_min_force_n", .05),
            origin="local MPPI cost, not original SPIDER or DexForge ForceAware",
            force_source="native solved per-finger object-contact normal force at each prediction step"))
        if keep_scale > 0:
            result["cost_environment_scope"] += "; local CA-finger contact-load deficit penalty enabled"
        for case in cases:
            case_root = (output/case["trajectory"]).parent
            for name in ("ca_input_hashes.json", "reference_manifest.json", "control_initialization.json", "contact_retention.json"):
                path = case_root/name
                if path.is_file():
                    result["input_sha256"][str(path)] = sha256(path)
    if config.task.get("options", {}).get("reference_method") == "dexforge_contactaware":
        manifest = ROOT / "third_party/DexForge/CONTACTAWARE_UPSTREAM.json"
        import json

        source = json.loads(manifest.read_text())
        result["contactaware"] = dict(
            repository=source["repository"], commit=source["commit"],
            source_manifest_sha256=sha256(manifest),
            adapter="single-key primitive contact anchors and WUJI2 IK; fixed object pose; no MANO frontend",
            reference="explicit contactaware hand_reference_qpos and active reference_contact_mask",
        )
    return result
