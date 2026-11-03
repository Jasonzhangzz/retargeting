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
            p.add_argument("--steps", type=int, default=3000)
            p.add_argument("--batch-size", type=int, default=256)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--validation-every", type=int, default=250)
            p.add_argument("--patience", type=int, default=6)
        elif command == "evaluate":
            p.add_argument("--seeds", type=csv_int, help="trained model seeds; default: --seed")
            p.add_argument("--noise-seeds", type=csv_int, default=(11, 23, 37))
            p.add_argument("--noise", type=csv_float, default=(0., .03, .1, .3),
                           help="comma-separated additive sigmas in normalized finger coordinates")
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
        config = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                             validation_every=args.validation_every, patience=args.patience)
        methods = ("dictionary", "diffusion") if args.model == "all" else (args.model,)
        results = []
        for seed in args.seeds or (args.seed,):
            for method in methods:
                results.append(train(data, method, seed, config, args.device))
        write_json(data.directory / "training_runs.json", results)
    elif args.command == "evaluate":
        result = evaluate(data, args.seeds or (args.seed,), args.noise_seeds, args.noise, args.device)
        print(json.dumps({k: result[k] for k in ("selected_method", "summaries")}, indent=2))
    else:
        from retarget.env.grasp.manifold_report import demo
        methods = METHODS if args.model == "all" else (args.model,)
        if args.model == "selected":
            methods = (json.loads((data.directory / "evaluation.json").read_text())["selected_method"],)
        path = demo(data, methods, args.seed, args.noise_seed, args.noise, args.tightening,
                    args.grasp_type, args.device)
        print(path)


if __name__ == "__main__":
    main()
