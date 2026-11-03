import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from mission_validation import validate_path
from mission_prompts import MISSION_PROMPTS, SYSTEM_PROMPT


def test_open_grid_all_objectives_have_exact_reference():
    grid = np.zeros((5, 5), dtype=np.uint8)
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for mission in ("efficiency", "safety", "smoothness", "multi"):
        report = validate_path(path, grid, (0, 0), (4, 4), mission)
        assert report.valid
        assert report.mission_objective_optimal
        assert abs(report.mission_objective_gap) < 1e-7


def test_safety_reference_finds_widest_route():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 3] = 1
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    report = validate_path(path, grid, (0, 0), (6, 0), "safety")
    assert report.valid
    assert report.mission_optimum_value == report.min_clearance
    assert report.mission_objective_optimal


def test_multi_reference_cost_is_not_less_than_geometric_optimum():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    path = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (4, 4)]
    report = validate_path(path, grid, (0, 0), (4, 4), "multi")
    assert report.valid
    assert report.mission_optimum_value <= report.multi_weighted_cost + 1e-9


def test_prompts_encode_state_and_bounds_contracts():
    assert "COMPLETE search state" in SYSTEM_PROMPT
    assert "state=(row,col,prev_dr,prev_dc)" not in MISSION_PROMPTS["smoothness"]
    assert "full-state dictionary keys" in SYSTEM_PROMPT.lower()
    assert "bounds" in MISSION_PROMPTS["multi"].lower()
    assert "step = sqrt(2) if dr != 0 and dc != 0 else 1.0" in MISSION_PROMPTS["efficiency"]
