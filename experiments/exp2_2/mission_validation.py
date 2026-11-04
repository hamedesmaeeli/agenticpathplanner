"""Independent path and mission-objective validation for Exp2.

The validator intentionally does not call the repository's evaluate_path().
It mirrors the repository's current movement convention: 8-connected moves,
Euclidean step lengths, and diagonal corner cutting permitted. This keeps the
validation semantics comparable with the current A* baseline in Exp1.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math
from typing import Iterable, Sequence

import numpy as np

Coord = tuple[int, int]
Path = Sequence[Coord]
MOVES: tuple[Coord, ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)
SAFETY_TARGET = 2.0
MULTI_CLEARANCE_WEIGHT = 0.5


@dataclass
class ValidationReport:
    mission: str
    valid: bool
    has_path: bool
    starts_correctly: bool
    ends_correctly: bool
    coordinates_well_formed: bool
    in_bounds: bool
    obstacle_free: bool
    adjacent_steps: bool
    corner_cutting_steps: int
    path_length: float
    euclidean_lower_bound: float
    plr_euclidean: float
    shortest_grid_path_length: float
    efficiency_ratio: float
    min_clearance: float
    mean_clearance: float
    safety_threshold: float
    safety_threshold_feasible: bool
    safety_threshold_met: bool
    total_turn_angle_rad: float
    turn_count: int
    multi_weighted_cost: float
    mission_objective_value: float
    mission_optimum_value: float
    mission_optimum_secondary_value: float
    mission_objective_gap: float
    mission_objective_ratio: float
    mission_objective_optimal: bool
    errors: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _coord(value: object) -> Coord | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    # bool is an int subclass but should not be accepted as a coordinate.
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in value):
        return None
    return int(value[0]), int(value[1])


def _clearance_map(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    obstacles = np.argwhere(grid == 1)
    if len(obstacles) == 0:
        return np.full((h, w), math.inf, dtype=float)
    result = np.empty((h, w), dtype=float)
    for r in range(h):
        for c in range(w):
            delta = obstacles - np.array([r, c])
            result[r, c] = float(np.sqrt((delta * delta).sum(axis=1)).min())
    return result


def _path_length(path: Sequence[Coord]) -> float:
    return sum(math.hypot(path[i + 1][0] - path[i][0],
                          path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def _turn_metrics(path: Sequence[Coord]) -> tuple[float, int]:
    total = 0.0
    count = 0
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        na, nb = math.hypot(*a), math.hypot(*b)
        if na == 0 or nb == 0:
            continue
        cosang = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
        angle = math.acos(cosang)
        # Suppress floating-point residue for exactly collinear vectors.
        if angle < 1e-7:
            angle = 0.0
        total += angle
        if angle > 1e-7:
            count += 1
    return total, count


def _corner_cutting_count(path: Sequence[Coord], grid: np.ndarray) -> int:
    """Count diagonal moves whose two orthogonal side cells are both blocked.

    This is diagnostic only: the current project A* allows these moves, so they
    are not considered invalid in Exp2's compatibility mode.
    """
    h, w = grid.shape
    count = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = r2 - r1, c2 - c1
        if abs(dr) == 1 and abs(dc) == 1:
            side1 = (r1 + dr, c1)
            side2 = (r1, c1 + dc)
            b1 = not (0 <= side1[0] < h and 0 <= side1[1] < w) or grid[side1] == 1
            b2 = not (0 <= side2[0] < h and 0 <= side2[1] < w) or grid[side2] == 1
            if b1 and b2:
                count += 1
    return count


def _shortest_grid_path_length(grid: np.ndarray, start: Coord, goal: Coord) -> float:
    """Dijkstra optimum under current 8-neighbor movement semantics."""
    h, w = grid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w
            and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return math.inf
    if grid[start] != 0 or grid[goal] != 0:
        return math.inf
    dist = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != dist.get(p):
            continue
        if p == goal:
            return cost
        for dr, dc in MOVES:
            q = (p[0] + dr, p[1] + dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w) or grid[q] != 0:
                continue
            nc = cost + math.hypot(dr, dc)
            if nc < dist.get(q, math.inf):
                dist[q] = nc
                heapq.heappush(queue, (nc, q))
    return math.inf


def _threshold_feasible(grid: np.ndarray, clearance: np.ndarray,
                        start: Coord, goal: Coord, threshold: float) -> bool:
    h, w = grid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w
            and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return False
    if grid[start] != 0 or grid[goal] != 0:
        return False
    if clearance[start] < threshold or clearance[goal] < threshold:
        return False
    seen = {start}
    q = deque([start])
    while q:
        p = q.popleft()
        if p == goal:
            return True
        for dr, dc in MOVES:
            n = (p[0] + dr, p[1] + dc)
            if (0 <= n[0] < h and 0 <= n[1] < w and n not in seen
                    and grid[n] == 0 and clearance[n] >= threshold):
                seen.add(n)
                q.append(n)
    return False


def _multi_cost(path: Sequence[Coord], clearance: np.ndarray) -> float:
    total = 0.0
    for u, v in zip(path, path[1:]):
        d = math.hypot(v[0] - u[0], v[1] - u[1])
        cv = float(clearance[v])
        risk = 0.0 if math.isinf(cv) else 1.0 / (1.0 + cv)
        total += d * (1.0 + MULTI_CLEARANCE_WEIGHT * risk)
    return total



def _safety_optimum(grid: np.ndarray, clearance: np.ndarray, start: Coord, goal: Coord) -> tuple[float, float]:
    """Maximum achievable bottleneck clearance; length breaks bottleneck ties."""
    h, w = grid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and
            0 <= goal[0] < h and 0 <= goal[1] < w):
        return (-math.inf, math.inf)
    if grid[start] != 0 or grid[goal] != 0:
        return (-math.inf, math.inf)
    best = {start: (float(clearance[start]), 0.0)}
    counter = 0
    queue = [(-best[start][0], 0.0, counter, start)]
    while queue:
        neg_bottleneck, length, _, p = heapq.heappop(queue)
        label = (-neg_bottleneck, length)
        if label != best.get(p):
            continue
        if p == goal:
            return label
        for dr, dc in MOVES:
            q = (p[0] + dr, p[1] + dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and grid[q] == 0:
                candidate = (min(label[0], float(clearance[q])),
                             label[1] + math.hypot(dr, dc))
                old = best.get(q, (-math.inf, math.inf))
                if candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    best[q] = candidate
                    counter += 1
                    heapq.heappush(queue, (-candidate[0], candidate[1], counter, q))
    return (-math.inf, math.inf)


def _smoothness_optimum(grid: np.ndarray, start: Coord, goal: Coord) -> tuple[float, float]:
    """Exact minimum total turning angle using direction-augmented states."""
    h, w = grid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and
            0 <= goal[0] < h and 0 <= goal[1] < w):
        return (math.inf, math.inf)
    if grid[start] != 0 or grid[goal] != 0:
        return (math.inf, math.inf)
    initial = (start[0], start[1], 0, 0)
    best = {initial: (0.0, 0.0)}
    queue = [(0.0, 0.0, 0, initial)]
    counter = 0
    best_goal = (math.inf, math.inf)
    while queue:
        angle_cost, length_cost, _, state = heapq.heappop(queue)
        if (angle_cost, length_cost) != best.get(state):
            continue
        r, c, prev_dr, prev_dc = state
        if (r, c) == goal:
            best_goal = min(best_goal, (angle_cost, length_cost))
            # Lexicographic Dijkstra: no later state can improve this pair.
            return (angle_cost, length_cost)
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] != 0:
                continue
            turn = 0.0
            if (prev_dr, prev_dc) != (0, 0):
                dot = prev_dr * dr + prev_dc * dc
                n1 = math.hypot(prev_dr, prev_dc)
                n2 = math.hypot(dr, dc)
                turn = math.acos(max(-1.0, min(1.0, dot / (n1 * n2))))
                if turn < 1e-7:
                    turn = 0.0
            step = math.hypot(dr, dc)
            nxt = (nr, nc, dr, dc)
            candidate = (angle_cost + turn, length_cost + step)
            if candidate < best.get(nxt, (math.inf, math.inf)):
                best[nxt] = candidate
                counter += 1
                heapq.heappush(queue, (candidate[0], candidate[1], counter, nxt))
    return best_goal


def _multi_optimum(grid: np.ndarray, clearance: np.ndarray, start: Coord, goal: Coord) -> float:
    """Exact Dijkstra optimum for the specified weighted edge objective."""
    h, w = grid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and
            0 <= goal[0] < h and 0 <= goal[1] < w):
        return math.inf
    if grid[start] != 0 or grid[goal] != 0:
        return math.inf
    dist = {start: 0.0}
    queue = [(0.0, 0, start)]
    counter = 0
    while queue:
        cost, _, p = heapq.heappop(queue)
        if cost != dist.get(p):
            continue
        if p == goal:
            return cost
        for dr, dc in MOVES:
            q = (p[0] + dr, p[1] + dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w) or grid[q] != 0:
                continue
            cv = float(clearance[q])
            risk = 0.0 if math.isinf(cv) else 1.0 / (1.0 + cv)
            candidate = cost + math.hypot(dr, dc) * (1.0 + MULTI_CLEARANCE_WEIGHT * risk)
            if candidate < dist.get(q, math.inf):
                dist[q] = candidate
                counter += 1
                heapq.heappush(queue, (candidate, counter, q))
    return math.inf

def validate_path(path: Iterable[object] | None, grid: np.ndarray,
                  start: Coord, goal: Coord, mission: str,
                  safety_target: float = SAFETY_TARGET) -> ValidationReport:
    """Validate a returned path and calculate independent mission metrics."""
    if mission not in {"efficiency", "safety", "smoothness", "multi"}:
        raise ValueError(f"Unknown mission: {mission}")
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be a 2-D array")
    h, w = grid.shape
    raw = list(path or [])
    converted: list[Coord] = []
    well_formed = True
    for item in raw:
        c = _coord(item)
        if c is None:
            well_formed = False
            break
        converted.append(c)
    p = converted if well_formed else []
    has_path = bool(raw)
    starts = bool(p) and p[0] == tuple(start)
    ends = bool(p) and p[-1] == tuple(goal)
    in_bounds = bool(p) and all(0 <= r < h and 0 <= c < w for r, c in p)
    obstacle_free = bool(p) and in_bounds and all(grid[r, c] == 0 for r, c in p)
    adjacent = bool(p) and all(
        max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1
        and (a != b)
        for a, b in zip(p, p[1:])
    )
    # A one-point path is valid only for start == goal.
    if len(p) == 1:
        adjacent = tuple(start) == tuple(goal) == p[0]
    errors: list[str] = []
    if not has_path: errors.append("empty_path")
    if not well_formed: errors.append("malformed_coordinate")
    if not starts: errors.append("wrong_start")
    if not ends: errors.append("wrong_goal")
    if not in_bounds: errors.append("out_of_bounds")
    if not obstacle_free: errors.append("obstacle_collision_or_invalid_cell")
    if not adjacent: errors.append("non_adjacent_waypoints")
    valid = bool(has_path and well_formed and starts and ends and in_bounds
                 and obstacle_free and adjacent)

    if valid:
        length = _path_length(p)
        obs_clearance = _clearance_map(grid)
        clearances = [float(obs_clearance[r, c]) for r, c in p]
        min_clear = min(clearances)
        mean_clear = (sum(clearances) / len(clearances))
        turn_angle, turn_count = _turn_metrics(p)
        multi = _multi_cost(p, obs_clearance)
        straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
        plr = length / straight if straight > 0 else (1.0 if length == 0 else math.inf)
        optimum = _shortest_grid_path_length(grid, tuple(start), tuple(goal))
        efficiency_ratio = (length / optimum if math.isfinite(optimum) and optimum > 0
                            else (1.0 if length == 0 and optimum == 0 else math.inf))
        threshold_feasible = _threshold_feasible(grid, obs_clearance, tuple(start), tuple(goal), safety_target)
        threshold_met = min_clear >= safety_target
        corner_count = _corner_cutting_count(p, grid)
    else:
        length = 0.0
        straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
        plr = math.inf
        optimum = _shortest_grid_path_length(grid, tuple(start), tuple(goal))
        efficiency_ratio = math.inf
        obs_clearance = _clearance_map(grid)
        min_clear = 0.0
        mean_clear = 0.0
        turn_angle = math.inf
        turn_count = 0
        multi = math.inf
        threshold_feasible = _threshold_feasible(grid, obs_clearance, tuple(start), tuple(goal), safety_target)
        threshold_met = False
        corner_count = 0

    if mission == "efficiency":
        objective = efficiency_ratio
        optimum_objective = 1.0 if math.isfinite(optimum) else math.inf
        optimum_secondary = optimum
        objective_gap = objective - optimum_objective
        objective_ratio = objective / optimum_objective if optimum_objective > 0 else math.inf
    elif mission == "safety":
        objective = min_clear
        optimum_objective, optimum_secondary = _safety_optimum(grid, obs_clearance, tuple(start), tuple(goal))
        objective_gap = (0.0 if objective == optimum_objective else optimum_objective - objective)
        objective_ratio = (1.0 if objective == optimum_objective == math.inf else
                           objective / optimum_objective if optimum_objective > 0 and math.isfinite(optimum_objective) else math.nan)
    elif mission == "smoothness":
        objective = turn_angle
        optimum_objective, optimum_secondary = _smoothness_optimum(grid, tuple(start), tuple(goal))
        objective_gap = objective - optimum_objective
        objective_ratio = (1.0 if objective == optimum_objective == 0 else
                           objective / optimum_objective if optimum_objective > 0 else math.nan)
    else:
        objective = multi
        optimum_objective = _multi_optimum(grid, obs_clearance, tuple(start), tuple(goal))
        optimum_secondary = math.nan
        objective_gap = objective - optimum_objective
        objective_ratio = objective / optimum_objective if optimum_objective > 0 and math.isfinite(optimum_objective) else math.inf
    if mission in ("safety", "smoothness"):
        primary_equal = (objective == optimum_objective if math.isinf(objective) or math.isinf(optimum_objective)
                         else abs(objective_gap) <= 1e-7 * max(1.0, abs(optimum_objective)))
        secondary_equal = abs(length - optimum_secondary) <= 1e-7 * max(1.0, abs(optimum_secondary))
        objective_optimal = bool(valid and primary_equal and secondary_equal)
    else:
        objective_optimal = bool(valid and math.isfinite(objective) and math.isfinite(optimum_objective)
                                 and abs(objective_gap) <= 1e-7 * max(1.0, abs(optimum_objective)))

    return ValidationReport(
        mission=mission, valid=valid, has_path=has_path,
        starts_correctly=starts, ends_correctly=ends,
        coordinates_well_formed=well_formed, in_bounds=in_bounds,
        obstacle_free=obstacle_free, adjacent_steps=adjacent,
        corner_cutting_steps=corner_count, path_length=length,
        euclidean_lower_bound=straight, plr_euclidean=plr,
        shortest_grid_path_length=optimum, efficiency_ratio=efficiency_ratio,
        min_clearance=min_clear, mean_clearance=mean_clear,
        safety_threshold=safety_target,
        safety_threshold_feasible=threshold_feasible,
        safety_threshold_met=threshold_met,
        total_turn_angle_rad=turn_angle, turn_count=turn_count,
        multi_weighted_cost=multi, mission_objective_value=objective,
        mission_optimum_value=optimum_objective, mission_optimum_secondary_value=optimum_secondary,
        mission_objective_gap=objective_gap,
        mission_objective_ratio=objective_ratio, mission_objective_optimal=objective_optimal,
        errors=errors,
    )
