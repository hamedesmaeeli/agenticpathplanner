"""CLI entry point: agentpp run / report / list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentpp",
                                     description="Agentic Path Planner — benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run benchmark experiments")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--model")
    run_p.add_argument("--mode", choices=["generate_path", "generate_algorithm"])
    run_p.add_argument("--output", default="results")
    run_p.add_argument("--no-baseline", action="store_true")

    rpt_p = sub.add_parser("report", help="Print results summary")
    rpt_p.add_argument("--results", default="results")

    lst_p = sub.add_parser("list", help="List maps and Ollama models")
    lst_p.add_argument("--maps",   action="store_true")
    lst_p.add_argument("--models", action="store_true")

    args = parser.parse_args()
    {"run": _run, "report": _report, "list": _list}[args.command](args)


def _run(args) -> None:
    import yaml
    from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
    from agenticpathplanner.classical.astar import AStarPlanner
    from agenticpathplanner.evaluation.evaluator import Evaluator
    from agenticpathplanner.llm.planner import LLMPlanner

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model  = args.model  or cfg.get("model",  "llama3.2")
    mode   = args.mode   or cfg.get("mode",   "generate_path")
    runs   = cfg.get("runs_per_map", 5)
    output = args.output or cfg.get("output_dir", "results")

    loader = CEC2024Loader()
    instances = [loader.load(m) for m in (cfg.get("maps") or loader.available_ids())]

    planners = []
    if not args.no_baseline:
        planners.append(AStarPlanner())
    planners.append(LLMPlanner(model=model, mode=mode))

    Evaluator(runs_per_map=runs, output_dir=output).run(planners, instances)


def _report(args) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    files = sorted(Path(args.results).glob("*.json"))
    if not files:
        console.print("[red]No result files found.")
        sys.exit(1)

    data = json.loads(files[-1].read_text())
    table = Table(title=files[-1].name)
    for col in ("Planner", "Map", "SR", "PLR", "SD", "SM"):
        table.add_column(col, justify="right" if col not in ("Planner", "Map") else "left")

    for planner, maps in data.items():
        for map_id, agg in maps.items():
            table.add_row(
                planner, map_id,
                f"{agg.get('sr',0):.0%}",
                f"{agg['plr_mean']:.3f}" if agg.get("plr_mean") else "—",
                f"{agg['sd_mean']:.2f}"  if agg.get("sd_mean")  else "—",
                f"{agg['sm_mean']:.2f}"  if agg.get("sm_mean")  else "—",
            )
    console.print(table)


def _list(args) -> None:
    from rich.console import Console
    from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
    from agenticpathplanner.llm.client import OllamaClient

    console = Console()
    if args.maps or not args.models:
        console.print("[bold]Maps:[/]")
        for mid in CEC2024Loader().available_ids():
            m = CEC2024Loader().load(mid)
            console.print(f"  {mid}  ({m.difficulty}, {m.height}×{m.width})")
    if args.models or not args.maps:
        console.print("[bold]Ollama models:[/]")
        for m in OllamaClient().list_models():
            console.print(f"  {m}")
