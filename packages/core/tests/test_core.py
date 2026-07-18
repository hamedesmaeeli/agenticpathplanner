"""Tests for agenticpathplanner.core (no external dependencies)."""

import numpy as np
import pytest

from agenticpathplanner.core import Coord, GridWorld, MapInstance, Path


def _simple_instance() -> MapInstance:
    g = np.zeros((10, 10), dtype=np.uint8)
    g[5, 1:4] = 1
    g[5, 6:9] = 1
    return MapInstance("test_simple", g, (1, 1), (8, 8))


class TestMapInstance:
    def test_is_free_on_open_cell(self):
        inst = _simple_instance()
        assert inst.is_free(0, 0)

    def test_is_free_on_obstacle(self):
        inst = _simple_instance()
        assert not inst.is_free(5, 2)

    def test_is_free_out_of_bounds(self):
        inst = _simple_instance()
        assert not inst.is_free(-1, 0)
        assert not inst.is_free(0, 99)

    def test_text_grid_has_S_and_G(self):
        txt = _simple_instance().as_text_grid()
        assert "S" in txt and "G" in txt

    def test_straight_line_distance(self):
        inst = _simple_instance()
        expected = float(np.sqrt((8 - 1) ** 2 + (8 - 1) ** 2))
        assert inst.straight_line_distance() == pytest.approx(expected)


class TestGridWorld:
    def test_neighbours_are_free(self):
        inst = _simple_instance()
        world = GridWorld(inst)
        for nb in world.neighbours(*inst.start):
            assert inst.is_free(*nb)

    def test_step_cost_diagonal(self):
        inst = _simple_instance()
        world = GridWorld(inst)
        assert world.step_cost((0, 0), (1, 1)) == pytest.approx(2 ** 0.5)

    def test_step_cost_cardinal(self):
        inst = _simple_instance()
        world = GridWorld(inst)
        assert world.step_cost((0, 0), (0, 1)) == pytest.approx(1.0)

    def test_heuristic_is_euclidean(self):
        inst = _simple_instance()
        world = GridWorld(inst)
        h = world.heuristic(inst.start)
        expected = inst.straight_line_distance()
        assert h == pytest.approx(expected)
