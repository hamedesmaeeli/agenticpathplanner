#!/usr/bin/env python3
"""End-to-end CEC2024 case study runner.

Runs all classical baselines and all three LLM agents on the three CEC2024
map instances, then produces side-by-side path visualisations and a
markdown comparison report.

Usage (classical-only, no Ollama required):
    python experiments/run_case_study.py --no-llm --output case_study/

Full run with Ollama:
    python experiments/run_case_study.py \\
        --output case_study/ \\
        --model-generator qwen2.5-coder:7b \\
        --model-refiner   llama3.1:8b \\
        --model-selector  mistral:7b

Output:
    case_study/
    ├── figures/
    │   ├── cec2024_s01.png
    │   ├── cec2024_m01.png
    │   └── cec2024_c01.png
    ├── results.json
    └── case_study_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sure the packages are importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "core"      / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "benchmarks"/ "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "classical" / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "llm"       / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "evaluation"/ "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "refiner"   / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "generator" / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "packages" / "selector"  / "src"))

from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
from agenticpathplanner.benchmarks.cec2024.metrics import evaluate_path, PathMetrics
from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.classical.theta_star import ThetaStarPlanner
from agenticpathplanner.classical.apf import APFRefiner
from agenticpathplanner.core.agent import AgentContext
from agenticpathplanner.core.types import MapInstance
from agenticpathplanner.evaluation.visualiser import render_comparison, save_report
from agenticpathplanner.selector.baselines import (
    RandomSelector, RuleBasedSelector, OracleSelector,
)
from agenticpathplanner.selector.registry import AlgorithmRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context builders — inject idea-specific fields into AgentContext
# ---------------------------------------------------------------------------

_MISSIONS = {
    "simple":  "Navigate from start to goal in a small open environment. Prioritise shortest path while maintaining safe clearance from walls.",
    "medium":  "Navigate a corridor-maze environment with moderate battery. Balance path length and obstacle clearance. Avoid tight corners.",
    "complex": "Navigate a dense cluttered environment under time pressure. Find any valid path quickly; safety margins may be reduced if needed.",
}

_ENVIRONMENTS = {
    "simple":  {"obstacle_density": "low",    "wind_speed": "5 knots N",  "urgency": "low"},
    "medium":  {"obstacle_density": "medium", "wind_speed": "12 knots NW","urgency": "medium"},
    "complex": {"obstacle_density": "high",   "wind_speed": "20 knots W", "urgency": "high"},
}


def _build_refiner_context(instance: MapInstance) -> AgentContext:
    initial_result = AStarPlanner().solve(instance)
    return AgentContext(
        instance=instance,
        initial_path=initial_result.path,
        environment=_ENVIRONMENTS[instance.difficulty],
        mission=_MISSIONS[instance.difficulty],
    )


def _build_generator_context(instance: MapInstance) -> AgentContext:
    return AgentContext(
        instance=instance,
        mission=_MISSIONS[instance.difficulty],
        environment=_ENVIRONMENTS[instance.difficulty],
    )


def _build_selector_context(instance: MapInstance) -> AgentContext:
    return AgentContext(
        instance=instance,
        environment=_ENVIRONMENTS[instance.difficulty],
    )


# ---------------------------------------------------------------------------
# Main case study logic
# ---------------------------------------------------------------------------

def run_case_study(
    output_dir: str,
    model_generator: str,
    model_refiner: str,
    model_selector: str,
    no_llm: bool = False,
    map_ids: list[str] | None = None,
) -> None:
    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    loader   = CEC2024Loader()
    ids      = map_ids or ["cec2024_s01", "cec2024_m01", "cec2024_c01"]
    instances = [loader.load(m) for m in ids]

    # Build registry
    registry = AlgorithmRegistry.default()

    # Classical planners (always run)
    classical_planners = [
        AStarPlanner(),
        ThetaStarPlanner(),
        APFRefiner(),
    ]

    # Selector baselines (always run)
    selector_baselines = [
        RandomSelector(registry, seed=42),
        RuleBasedSelector(registry),
    ]

    # LLM agents (skipped with --no-llm)
    agents: list = []
    if not no_llm:
        from agenticpathplanner.refiner.agent import RefinerAgent
        from agenticpathplanner.generator.agent import GeneratorAgent
        from agenticpathplanner.selector.agent import SelectorAgent
        agents = [
            RefinerAgent(model=model_refiner),
            GeneratorAgent(model=model_generator),
            SelectorAgent(model=model_selector, registry=registry),
        ]

    all_results:  dict[str, dict[str, PathMetrics]] = {}
    all_paths:    dict[str, dict[str, list]]         = {}
    oracle_bests: dict[str, str]                     = {}

    for instance in instances:
        log.info("── Map: %s (%s) ──", instance.map_id, instance.difficulty)
        paths:   dict[str, list]        = {}
        metrics: dict[str, PathMetrics] = {}

        # 1. Classical planners
        for planner in classical_planners:
            log.info("  Running %s …", planner.name)
            result = planner.solve(instance)
            paths[planner.name]   = result.path
            metrics[planner.name] = evaluate_path(result.path, instance.grid, instance.start, instance.goal)
            _log_metrics(planner.name, metrics[planner.name])

        # 2. Selector baselines
        for sel in selector_baselines:
            log.info("  Running %s …", sel.name)
            result = sel.solve(instance)
            paths[sel.name]   = result.path
            metrics[sel.name] = evaluate_path(result.path, instance.grid, instance.start, instance.goal)
            _log_metrics(sel.name, metrics[sel.name])

        # 3. LLM agents
        for agent in agents:
            log.info("  Running %s …", agent.name)
            if "refiner" in agent.name:
                ctx = _build_refiner_context(instance)
            elif "generator" in agent.name:
                ctx = _build_generator_context(instance)
            else:
                ctx = _build_selector_context(instance)
            result = agent.run(ctx)
            paths[agent.name]   = result.path
            metrics[agent.name] = evaluate_path(result.path, instance.grid, instance.start, instance.goal)
            _log_metrics(agent.name, metrics[agent.name])

        # Determine oracle best (lowest PLR among valid paths)
        valid_methods = {n: m for n, m in metrics.items() if m.valid}
        if valid_methods:
            oracle_bests[instance.map_id] = min(valid_methods, key=lambda n: valid_methods[n].plr)

        all_paths[instance.map_id]   = paths
        all_results[instance.map_id] = metrics

        # Render comparison figure
        fig_path = out / "figures" / f"{instance.map_id}.png"
        render_comparison(instance, paths, metrics, save_path=fig_path)
        log.info("  Figure saved → %s", fig_path)

    # Add Oracle selector results (post-hoc)
    if oracle_bests:
        oracle = OracleSelector(registry, oracle_map=oracle_bests)
        for instance in instances:
            result = oracle.solve(instance)
            all_paths[instance.map_id]["oracle_selector"]   = result.path
            all_results[instance.map_id]["oracle_selector"] = evaluate_path(
                result.path, instance.grid, instance.start, instance.goal
            )

    # Save results.json
    json_out = out / "results.json"
    serialisable = {
        map_id: {name: m.as_dict() for name, m in methods.items()}
        for map_id, methods in all_results.items()
    }
    json_out.write_text(json.dumps(serialisable, indent=2))
    log.info("Results saved → %s", json_out)

    # Save case study report
    instance_meta = {inst.map_id: inst for inst in instances}
    save_report(
        all_results,
        figures_dir=out / "figures",
        output_md=out / "case_study_report.md",
        instance_meta=instance_meta,
    )

    # Print summary table to console
    _print_summary(all_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_metrics(name: str, m: PathMetrics) -> None:
    if m.valid:
        log.info("    %-26s  PLR=%.3f  SD=%.2f  SM=%.2f", name, m.plr, m.sd, m.sm)
    else:
        log.info("    %-26s  INVALID", name)


def _print_summary(all_results: dict[str, dict[str, PathMetrics]]) -> None:
    print("\n" + "="*70)
    print("CASE STUDY SUMMARY")
    print("="*70)
    for map_id, methods in all_results.items():
        print(f"\n{map_id}")
        print(f"  {'Method':<30} {'Valid':>5} {'PLR':>7} {'SD':>6} {'SM':>6}")
        print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*6} {'-'*6}")
        for name, m in methods.items():
            if m.valid:
                print(f"  {name:<30} {'✓':>5} {m.plr:>7.3f} {m.sd:>6.2f} {m.sm:>6.2f}")
            else:
                print(f"  {name:<30} {'✗':>5} {'—':>7} {'—':>6} {'—':>6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CEC2024 end-to-end case study runner")
    p.add_argument("--output",          default="case_study",    help="Output directory")
    p.add_argument("--model-generator", default="qwen2.5-coder:7b")
    p.add_argument("--model-refiner",   default="llama3.1:8b")
    p.add_argument("--model-selector",  default="mistral:7b")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM agents (classical baselines only — no Ollama needed)")
    p.add_argument("--maps", nargs="+",
                   default=["cec2024_s01", "cec2024_m01", "cec2024_c01"],
                   help="Map IDs to evaluate")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_case_study(
        output_dir=args.output,
        model_generator=args.model_generator,
        model_refiner=args.model_refiner,
        model_selector=args.model_selector,
        no_llm=args.no_llm,
        map_ids=args.maps,
    )
