"""Tests for agenticpathplanner.evaluation."""

import tempfile
from pathlib import Path

from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.evaluation.evaluator import Evaluator


class TestEvaluator:
    def test_run_produces_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = Evaluator(runs_per_map=1, output_dir=tmp)
            inst = BUILTIN_MAPS["cec2024_s01"]
            results = ev.run(planners=[AStarPlanner()], instances=[inst])
            assert "astar" in results
            assert "cec2024_s01" in results["astar"]
            assert results["astar"]["cec2024_s01"]["sr"] == 1.0

    def test_run_saves_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = Evaluator(runs_per_map=1, output_dir=tmp)
            ev.run([AStarPlanner()], [BUILTIN_MAPS["cec2024_s01"]])
            files = list(Path(tmp).iterdir())
            assert any(f.suffix == ".csv" for f in files)
            assert any(f.suffix == ".json" for f in files)
