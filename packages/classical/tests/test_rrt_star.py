"""Tests for RRTStarPlanner."""

import numpy as np
import pytest

from agenticpathplanner.classical.rrt_star import RRTStarPlanner
from agenticpathplanner.core.types import MapInstance


def _inst(grid, start=(0, 0), goal=None):
    g = goal or (grid.shape[0] - 1, grid.shape[1] - 1)
    return MapInstance("t", grid, start, g)


class TestRRTStarPlanner:
    def test_finds_path_open_grid(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        result = RRTStarPlanner(seed=1).solve(_inst(grid))
        assert result.success
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (9, 9)

    def test_path_collision_free(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[3:7, 5] = 1   # vertical wall
        result = RRTStarPlanner(seed=42).solve(_inst(grid, (0, 0), (9, 9)))
        if result.success:
            for r, c in result.path:
                assert grid[r, c] == 0, f"Path hit obstacle at ({r},{c})"

    def test_no_path_fully_blocked(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[:, 2] = 1   # full vertical wall — no gap
        result = RRTStarPlanner(max_iter=200, seed=0).solve(_inst(grid, (0, 0), (0, 4)))
        assert not result.success

    def test_registered_in_default_registry(self):
        from agenticpathplanner.selector.registry import AlgorithmRegistry
        reg = AlgorithmRegistry.default()
        assert reg.get("rrt_star") is not None

    def test_simple_benchmark_map(self):
        from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
        inst = BUILTIN_MAPS["cec2024_s01"]
        result = RRTStarPlanner(max_iter=3000, seed=7).solve(inst)
        assert result.success
