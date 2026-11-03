#!/usr/bin/env python3
"""Prepare, train, evaluate and inspect the isolated grasp-manifold experiment."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retarget.env.grasp.manifold import (ExperimentData, METHODS, TrainConfig, evaluate,
                                         prepare, train, write_json)
from retarget.env.grasp.manifold_flow import FlowConfig


def csv_int(value):
    return tuple(int(x) for x in value.split(","))


def csv_float(value):
    return tuple(float(x) for x in value.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "evaluate", "demo"):
        p = subparsers.add_parser(command)
        destination = "output" if command in ("prepare", "train") else "results"
        p.add_argument("--output", type=Path, default=ROOT / destination / "grasp_manifold_20261103")
        p.add_argument("--seed", type=int, default=7 if command in ("prepare", "train") else None,
                       help="model seed; evaluation/demo default to retained validation-selected checkpoints")
        p.add_argument("--device", default="auto")
        if command == "prepare":
            p.add_argument("--source", type=Path, default=ROOT / "results/grasp_current_20261103")
        elif command == "train":
            p.add_argument("--model", choices=("dictionary", "diffusion", "all"), default="all")
            p.add_argument("--seeds", type=csv_int, help="comma-separated model initialization seeds; default: --seed")
            p.add_argument("--steps", type=int, help="per stage; default: flow 10000, dictionary 3000")
            p.add_argument("--batch-size", type=int, default=256)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--validation-every", type=int, default=250)
            p.add_argument("--patience", type=int, default=12)
            p.add_argument("--latent-kind", choices=("vq", "vae", "both"), default="vq")
            p.add_argument("--stage", choices=("autoencoder", "flow", "all"), default="all")
            p.add_argument("--autoencoder-checkpoint", type=Path)
            p.add_argument("--min-steps", type=int, default=2500)
            for key in ("latent_dim", "codebook_size", "hidden_dim", "flow_steps", "max_rounds"):
                p.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(FlowConfig(), key))
            for key in ("sigma_min", "sigma_max", "reconstruction_weight", "cross_size_weight",
                        "consistency_weight", "contrastive_weight", "contrastive_temperature", "kl_weight",
                        "codebook_weight", "commitment_weight"):
                p.add_argument("--" + key.replace("_", "-"), type=float, default=getattr(FlowConfig(), key))
        elif command == "evaluate":
            p.add_argument("--baseline", type=Path, default=ROOT / "results/grasp_manifold_20261103",
                           help="retained legacy experiment for paired flow comparisons")
            p.add_argument("--seeds", type=csv_int, help="trained model seeds; default: --seed")
            p.add_argument("--noise-seeds", type=csv_int, default=(11, 23, 37))
            p.add_argument("--noise", type=csv_float,
                           help="normalized sigmas: per-round for flow (default 0,.03,.1,trained sigma-max); additive for legacy (0,.03,.1,.3)")
        else:
            p.add_argument("--model", choices=(*METHODS, "all", "selected"), default="all")
            p.add_argument("--grasp-type", type=int, help="GRASP taxonomy id; default: all supported types")
            p.add_argument("--noise", type=csv_float, default=(0., .03, .1))
            p.add_argument("--tightening", type=csv_float, default=(0., .1, .2))
            p.add_argument("--noise-seed", type=int, default=11)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output, args.seed)
        return
    data = ExperimentData(args.output)
    if args.command == "train":
        methods = ("dictionary", "diffusion") if args.model == "all" else (args.model,)
        results = []
        for seed in args.seeds or (args.seed,):
            for method in methods:
                if method == "dictionary":
                    config = TrainConfig(steps=args.steps if args.steps is not None else 3000, batch_size=args.batch_size, lr=args.lr,
                                         validation_every=args.validation_every, patience=args.patience)
                    results.append(train(data, method, seed, config, args.device))
                else:
                    config = FlowConfig(**{key: getattr(args, key) for key in FlowConfig.__dataclass_fields__
                                           if key != "steps"}, steps=args.steps if args.steps is not None else 10000)
                    kinds = ("vq", "vae") if args.latent_kind == "both" else (args.latent_kind,)
                    if args.stage == "flow" and (len(kinds) > 1 or len(args.seeds or (args.seed,)) > 1):
                        raise ValueError("Independent flow training needs one kind/seed per checkpoint")
                    for kind in kinds:
                        results.append(train(data, method, seed, config, args.device, latent_kind=kind,
                                             stage=args.stage, autoencoder_path=args.autoencoder_checkpoint))
                write_json(data.directory / "training_runs.json", results)
    elif args.command == "evaluate":
        if (data.directory / "models.json").exists():
            from retarget.env.grasp.manifold_flow_evaluation import evaluate_flow
            result = evaluate_flow(data, args.baseline, args.seeds or ((args.seed,) if args.seed is not None else None),
                                   args.noise_seeds, args.device, sigmas=args.noise)
        else:
            result = evaluate(data, args.seeds or (args.seed,), args.noise_seeds,
                              args.noise if args.noise is not None else (0., .03, .1, .3), args.device)
        print(json.dumps({k: result[k] for k in ("selected_method", "summaries")}, indent=2))
    else:
        from retarget.env.grasp.manifold_report import demo
        methods = data.available_methods() if args.model == "all" else (args.model,)
        if args.model == "selected":
            methods = (json.loads((data.directory / "evaluation.json").read_text())["selected_method"],)
        path = demo(data, methods, args.seed, args.noise_seed, args.noise, args.tightening,
                    args.grasp_type, args.device)
        print(path)


if __name__ == "__main__":
    main()
