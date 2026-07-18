"""Tests for agenticpathplanner.classical."""

import numpy as np
import pytest

from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.core.types import MapInstance


def _inst(grid, start=(0,0), goal=None):
    g = goal or (grid.shape[0]-1, grid.shape[1]-1)
    return MapInstance("t", grid, start, g)


class TestAStarPlanner:
    def test_finds_path_open_grid(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        inst = _inst(grid)
        result = AStarPlanner().solve(inst)
        assert result.success
        assert result.path[0] == inst.start
        assert result.path[-1] == inst.goal

    def test_no_path_blocked_goal(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[4, 4] = 1
        result = AStarPlanner().solve(_inst(grid))
        assert not result.success

    def test_path_avoids_obstacles(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[0:4, 2] = 1   # vertical wall with gap at row 4
        result = AStarPlanner().solve(_inst(grid, (0,0), (0,4)))
        assert result.success
        path_set = set(result.path)
        for r in range(4):
            assert (r, 2) not in path_set

    def test_medium_map(self):
        from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
        inst = BUILTIN_MAPS["cec2024_m01"]
        result = AStarPlanner().solve(inst)
        assert result.success
