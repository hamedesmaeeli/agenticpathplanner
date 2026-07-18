"""MPC path planner — receding-horizon optimisation with safety-weighted cost.

Implements the MPC predict–optimise–act cycle on 2-D occupancy grids:
  1. From the current position, run A* with a clearance-penalised cost function.
  2. Execute only the first step of the returned path.
  3. Replan from the new position and repeat.

The safety cost penalises proximity to obstacles, causing the planner to
route away from walls even at the cost of a longer path. The clearance
penalty map is precomputed once via a BFS distance transform, so replanning
at each step remains fast.

Reference: Camacho & Bordons (2004). Model Predictive Control. Springer.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import Coord, MapInstance, Path

_MOVES = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


class MPCPlanner(BasePlanner):
    """Receding-horizon MPC path planner with safety-weighted cost.

    At each position the full map is replanned with a cost function that
    penalises proximity to obstacles:
        g(next) += step_cost + clearance_weight / (dist_to_obstacle + 0.5)

    This naturally trades longer paths for greater clearance, complementing
    A* (pure length) and Theta* (smoothness).
    """

    name = "mpc"

    def __init__(self, clearance_weight: float = 2.0, max_steps: int = 500):
        self.clearance_weight = clearance_weight
        self.max_steps = max_steps

    def _plan(self, instance: MapInstance) -> Path:
        grid, start, goal = instance.grid, instance.start, instance.goal

        penalty_map = _build_penalty_map(grid, self.clearance_weight)

        current = start
        path: Path = [start]
        visited: set[Coord] = {start}

        for _ in range(self.max_steps):
            if current == goal:
                break
            nxt = _mpc_step(current, goal, grid, penalty_map, visited)
            if nxt is None:
                break
            current = nxt
            path.append(current)
            visited.add(current)

        return path if path[-1] == goal else []


# ---------------------------------------------------------------------------
# MPC step: safety-weighted A* returning only the first move
# ---------------------------------------------------------------------------

def _mpc_step(
    start: Coord,
    goal: Coord,
    grid: np.ndarray,
    penalty_map: np.ndarray,
    visited: set[Coord],
) -> Coord | None:
    """Run safety-weighted A* from start; return the first step toward goal."""
    H, W = grid.shape

    def h(p: Coord) -> float:
        return math.sqrt((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2)

    # heap: (f, g, node, first_step)
    heap: list = [(h(start), 0.0, start, None)]
    best_g: dict[Coord, float] = {start: 0.0}

    while heap:
        _, g_cur, cur, first = heapq.heappop(heap)

        if cur == goal:
            return first

        if g_cur > best_g.get(cur, float("inf")):
            continue

        for dr, dc in _MOVES:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == 1:
                continue
            nb: Coord = (nr, nc)
            if nb in visited and nb != goal:
                continue

            step_cost = math.sqrt(dr * dr + dc * dc)
            new_g = g_cur + step_cost + penalty_map[nr, nc]

            if new_g < best_g.get(nb, float("inf")):
                best_g[nb] = new_g
                first_step = first if first is not None else nb
                heapq.heappush(heap, (new_g + h(nb), new_g, nb, first_step))

    return None


# ---------------------------------------------------------------------------
# Clearance penalty map — BFS distance transform from all obstacles
# ---------------------------------------------------------------------------

def _build_penalty_map(grid: np.ndarray, clearance_weight: float) -> np.ndarray:
    """Precompute per-cell clearance penalty using a BFS distance transform.

    Cells close to obstacles receive a higher penalty so the planner routes
    away from walls without per-node distance queries.
    """
    H, W = grid.shape
    dist = np.full((H, W), float("inf"))
    q: deque[Coord] = deque()

    for r, c in np.argwhere(grid == 1):
        dist[r, c] = 0.0
        q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            d = dist[r, c] + math.sqrt(dr * dr + dc * dc)
            if d < dist[nr, nc]:
                dist[nr, nc] = d
                q.append((nr, nc))

    penalty = clearance_weight / (dist + 0.5)
    penalty[grid == 1] = 0.0  # blocked cells need no penalty
    return penalty
