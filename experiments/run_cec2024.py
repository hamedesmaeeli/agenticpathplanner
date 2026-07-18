#!/usr/bin/env python3
"""CEC2024 benchmark runner.

Usage:
    python experiments/run_cec2024.py --config experiments/configs/cec2024.yaml
    python experiments/run_cec2024.py --model codellama --mode generate_algorithm
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import yaml

from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.evaluation.evaluator import Evaluator
from agenticpathplanner.llm.planner import LLMPlanner

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/cec2024.yaml")
    p.add_argument("--model")
    p.add_argument("--mode", choices=["generate_path", "generate_algorithm"])
    p.add_argument("--output", default="results")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--difficulty", choices=["simple", "medium", "complex"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model  = args.model  or cfg.get("model",  "llama3.2")
    mode   = args.mode   or cfg.get("mode",   "generate_path")
    runs   = cfg.get("runs_per_map", 5)
    output = args.output or cfg.get("output_dir", "results")

    loader = CEC2024Loader()
    instances = []
    for mid in (cfg.get("maps") or loader.available_ids()):
        try:
            inst = loader.load(mid)
            if not args.difficulty or inst.difficulty == args.difficulty:
                instances.append(inst)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)

    if not instances:
        print("No instances.", file=sys.stderr); sys.exit(1)

    print(f"Maps: {[i.map_id for i in instances]}")
    print(f"Model: {model}  |  Mode: {mode}  |  Runs/map: {runs}\n")

    planners = []
    if not args.no_baseline:
        planners.append(AStarPlanner())
    planners.append(LLMPlanner(model=model, mode=mode,
                               temperature=cfg.get("temperature", 0.2),
                               include_grid=cfg.get("include_grid_in_prompt", True)))

    Evaluator(runs_per_map=runs, output_dir=output).run(planners, instances)


if __name__ == "__main__":
    main()
