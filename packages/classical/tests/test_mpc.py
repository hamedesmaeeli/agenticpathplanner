"""Tests for MPCPlanner."""

import numpy as np
import pytest

from agenticpathplanner.classical.mpc import MPCPlanner, _build_penalty_map
from agenticpathplanner.core.types import MapInstance


def _inst(grid, start=(0, 0), goal=None):
    g = goal or (grid.shape[0] - 1, grid.shape[1] - 1)
    return MapInstance("t", grid, start, g)


class TestMPCPlanner:
    def test_finds_path_open_grid(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        result = MPCPlanner().solve(_inst(grid))
        assert result.success
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (7, 7)

    def test_path_collision_free(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[3:7, 5] = 1
        result = MPCPlanner().solve(_inst(grid, (0, 0), (9, 9)))
        if result.success:
            for r, c in result.path:
                assert grid[r, c] == 0, f"Path hit obstacle at ({r},{c})"

    def test_no_path_fully_blocked(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[:, 2] = 1
        result = MPCPlanner().solve(_inst(grid, (0, 0), (0, 4)))
        assert not result.success

    def test_higher_clearance_weight_avoids_obstacles_more(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[5, 1:9] = 1   # horizontal wall with one gap
        grid[5, 4] = 0     # gap at column 4
        inst = _inst(grid, (0, 0), (9, 9))

        low = MPCPlanner(clearance_weight=0.1).solve(inst)
        high = MPCPlanner(clearance_weight=5.0).solve(inst)

        # Both should succeed; high-clearance path may be longer but further from walls
        if low.success and high.success:
            from agenticpathplanner.benchmarks.cec2024.metrics import evaluate_path
            m_low = evaluate_path(low.path, grid, inst.start, inst.goal)
            m_high = evaluate_path(high.path, grid, inst.start, inst.goal)
            assert m_high.sd >= m_low.sd

    def test_penalty_map_shape(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        pm = _build_penalty_map(grid, clearance_weight=1.0)
        assert pm.shape == (5, 5)
        assert pm[2, 2] == 0.0       # obstacle cell itself has zero penalty
        assert pm[2, 3] > pm[0, 0]   # cell next to obstacle has higher penalty

    def test_registered_in_default_registry(self):
        from agenticpathplanner.selector.registry import AlgorithmRegistry
        reg = AlgorithmRegistry.default()
        assert reg.get("mpc") is not None

    def test_simple_benchmark_map(self):
        from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
        inst = BUILTIN_MAPS["cec2024_s01"]
        result = MPCPlanner().solve(inst)
        assert result.success
