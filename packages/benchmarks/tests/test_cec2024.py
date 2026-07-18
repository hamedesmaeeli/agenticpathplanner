"""Tests for agenticpathplanner.benchmarks (no Ollama required)."""

import numpy as np
import pytest

from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
from agenticpathplanner.benchmarks.cec2024.metrics import (
    PathMetrics, aggregate_runs, evaluate_path,
)
from agenticpathplanner.core.types import MapInstance


class TestBuiltinMaps:
    def test_all_three_load(self):
        loader = CEC2024Loader()
        for mid in ("cec2024_s01", "cec2024_m01", "cec2024_c01"):
            inst = loader.load(mid)
            assert inst.map_id == mid
            assert inst.grid.ndim == 2

    def test_start_and_goal_free(self):
        for inst in BUILTIN_MAPS.values():
            assert inst.is_free(*inst.start), f"{inst.map_id}: start blocked"
            assert inst.is_free(*inst.goal),  f"{inst.map_id}: goal blocked"

    def test_unknown_map_raises(self):
        with pytest.raises(FileNotFoundError):
            CEC2024Loader().load("does_not_exist")


class TestMetrics:
    def test_empty_path_invalid(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        m = evaluate_path([], inst.grid, inst.start, inst.goal)
        assert not m.valid

    def test_obstacle_crossing_invalid(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        m = evaluate_path([(0,0),(2,2),(4,4)], grid, (0,0), (4,4))
        assert not m.valid

    def test_aggregate_sr(self):
        runs = [
            PathMetrics(1.5, 1.0, 0.5, True,  10),
            PathMetrics(0.0, 0.0, 0.0, False,  0),
            PathMetrics(1.2, 2.0, 0.3, True,   9),
        ]
        agg = aggregate_runs(runs)
        assert agg["sr"] == pytest.approx(2/3, rel=1e-3)
        assert agg["plr_mean"] == pytest.approx(1.35, rel=1e-3)
