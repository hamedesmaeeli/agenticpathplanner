"""Multi-run experiment evaluator — works with any BasePlanner subclass."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path as FSPath
from typing import Sequence

from rich.console import Console

from agenticpathplanner.benchmarks.cec2024.metrics import evaluate_path, aggregate_runs
from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import MapInstance

log = logging.getLogger(__name__)
console = Console()


class Evaluator:
    """Run planners over benchmark instances and aggregate results."""

    def __init__(self, runs_per_map: int = 5, output_dir: str | FSPath = "results"):
        self.runs_per_map = runs_per_map
        self.output_dir = FSPath(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        planners: Sequence[BasePlanner],
        instances: Sequence[MapInstance],
    ) -> dict:
        all_results: dict = {}
        for planner in planners:
            console.rule(f"[bold blue]Planner: {planner.name}")
            planner_results: dict = {}
            for instance in instances:
                metrics_list = []
                for _ in range(self.runs_per_map):
                    result = planner.solve(instance)
                    metrics_list.append(
                        evaluate_path(result.path, instance.grid, instance.start, instance.goal)
                    )
                agg = aggregate_runs(metrics_list)
                agg.update({"map_id": instance.map_id, "difficulty": instance.difficulty,
                             "planner": planner.name})
                planner_results[instance.map_id] = agg
                sr = agg.get("sr", 0.0)
                plr = agg.get("plr_mean")
                console.print(
                    f"  [green]{instance.map_id}[/]  SR={sr:.0%}"
                    + (f"  PLR={plr:.3f}  SD={agg['sd_mean']:.2f}  SM={agg['sm_mean']:.2f}"
                       if plr is not None else "  no valid paths")
                )
            all_results[planner.name] = planner_results
        self._save(all_results)
        return all_results

    def _save(self, all_results: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows = [agg for maps in all_results.values() for agg in maps.values()]
        if not rows:
            return
        csv_path = self.output_dir / f"results_{ts}.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        json_path = self.output_dir / f"results_{ts}.json"
        with json_path.open("w") as fh:
            json.dump(all_results, fh, indent=2)
        console.print(f"\n[dim]Saved → {csv_path.name}  {json_path.name}[/]")
