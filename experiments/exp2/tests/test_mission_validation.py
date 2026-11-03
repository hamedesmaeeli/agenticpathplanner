from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from mission_validation import validate_path
from mission_prompts import MISSION_PROMPTS, SYSTEM_PROMPT, PromptInput, build_user_prompt


def test_valid_straight_path_and_efficiency_optimum():
    grid = np.zeros((5, 5), dtype=np.uint8)
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    r = validate_path(path, grid, (0, 0), (4, 4), "efficiency")
    assert r.valid
    assert abs(r.path_length - 4 * np.sqrt(2)) < 1e-9
    assert abs(r.efficiency_ratio - 1.0) < 1e-9


def test_rejects_obstacle_and_non_adjacent_jump():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    blocked = validate_path([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                            grid, (0, 0), (4, 4), "efficiency")
    assert not blocked.valid
    assert "obstacle_collision_or_invalid_cell" in blocked.errors
    jump = validate_path([(0, 0), (4, 4)], grid, (0, 0), (4, 4), "efficiency")
    assert not jump.valid
    assert "non_adjacent_waypoints" in jump.errors


def test_smoothness_counts_direction_changes():
    grid = np.zeros((5, 5), dtype=np.uint8)
    straight = validate_path([(0, 0), (1, 1), (2, 2), (3, 3)],
                             grid, (0, 0), (3, 3), "smoothness")
    bent = validate_path([(0, 0), (1, 1), (1, 2), (2, 3)],
                         grid, (0, 0), (2, 3), "smoothness")
    assert straight.valid and bent.valid
    assert abs(straight.total_turn_angle_rad) < 1e-9
    assert bent.total_turn_angle_rad > 0.0
    assert bent.turn_count > 0


def test_safety_clearance_and_feasibility():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 3] = 1
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    r = validate_path(path, grid, (0, 0), (6, 0), "safety", safety_target=2.0)
    assert r.valid
    assert r.min_clearance >= 3.0
    assert r.safety_threshold_feasible
    assert r.safety_threshold_met


def test_multi_cost_matches_defined_edge_formula():
    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[1, 2] = 1
    path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    r = validate_path(path, grid, (0, 0), (3, 3), "multi")
    assert r.valid
    assert r.multi_weighted_cost >= r.path_length


def test_all_four_prompt_specs_are_distinct_and_built():
    assert set(MISSION_PROMPTS) == {"efficiency", "safety", "smoothness", "multi"}
    assert "octile" in MISSION_PROMPTS["efficiency"].lower()
    assert "bottleneck" in MISSION_PROMPTS["safety"].lower()
    assert "turning angle" in MISSION_PROMPTS["smoothness"].lower()
    assert "edge_cost" in MISSION_PROMPTS["multi"]
    data = PromptInput("m", 5, 5, "simple", (0, 0), (4, 4), 0.1, "S .\n. G")
    prompt = build_user_prompt(data, "efficiency")
    assert "plan(grid, start, goal)" in SYSTEM_PROMPT
    assert "octile" in prompt.lower()
    assert "S ." in prompt
