"""RRT* — asymptotically optimal sampling-based path planner.

Builds a tree of collision-free edges by randomly sampling the grid space,
steering toward samples, and rewiring nearby nodes to maintain near-optimal
costs. Unlike RRT (feasibility only), RRT* converges to the optimal path as
the number of iterations grows.

Reference: Karaman & Frazzoli (2011). Sampling-based algorithms for optimal
motion planning. IJRR 30(7), 846–894.
"""

from __future__ import annotations

import math

import numpy as np

from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import Coord, MapInstance, Path

_MOVES = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


class RRTStarPlanner(BasePlanner):
    """RRT* — asymptotically optimal sampling-based planner on 2-D occupancy grids."""

    name = "rrt_star"

    def __init__(
        self,
        max_iter: int = 3000,
        step_size: float = 2.0,
        goal_bias: float = 0.15,
        rewire_radius: float = 3.5,
        seed: int = 0,
    ):
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.seed = seed

    def _plan(self, instance: MapInstance) -> Path:
        grid, start, goal = instance.grid, instance.start, instance.goal
        H, W = grid.shape
        rng = np.random.default_rng(self.seed)

        parent: dict[Coord, Coord | None] = {start: None}
        cost: dict[Coord, float] = {start: 0.0}

        for _ in range(self.max_iter):
            # Sample — bias toward goal with probability goal_bias
            if rng.random() < self.goal_bias:
                sample: Coord = goal
            else:
                sample = (int(rng.integers(0, H)), int(rng.integers(0, W)))

            if grid[sample[0], sample[1]] == 1:
                continue

            # Nearest node in tree
            nearest = min(parent.keys(), key=lambda n: _dist(n, sample))

            # Steer toward sample by at most step_size
            new_node = _steer(nearest, sample, self.step_size, H, W)
            if grid[new_node[0], new_node[1]] == 1:
                continue
            if new_node in parent:  # already in tree — skip to avoid rewiring cycles
                continue
            if not _los(nearest, new_node, grid):
                continue

            # Near nodes within rewire radius
            near_nodes = [n for n in parent if _dist(n, new_node) <= self.rewire_radius]

            # Choose lowest-cost parent from near nodes
            best_parent = nearest
            best_cost = cost[nearest] + _dist(nearest, new_node)
            for n in near_nodes:
                if _los(n, new_node, grid):
                    c = cost[n] + _dist(n, new_node)
                    if c < best_cost:
                        best_cost = c
                        best_parent = n

            parent[new_node] = best_parent
            cost[new_node] = best_cost

            # Rewire: check if routing through new_node reduces cost for near nodes
            for n in near_nodes:
                if n == new_node:
                    continue
                if _los(new_node, n, grid):
                    c = cost[new_node] + _dist(new_node, n)
                    if c < cost.get(n, float("inf")):
                        parent[n] = new_node
                        cost[n] = c

            # Connect to goal if reachable from new_node
            if _dist(new_node, goal) <= self.step_size and _los(new_node, goal, grid):
                gc = cost[new_node] + _dist(new_node, goal)
                if gc < cost.get(goal, float("inf")):
                    parent[goal] = new_node
                    cost[goal] = gc

        if goal not in parent:
            return []

        # Reconstruct with a seen-set guard against accidental cycles
        path: Path = []
        node: Coord | None = goal
        seen: set[Coord] = set()
        while node is not None and node not in seen:
            seen.add(node)
            path.append(node)
            node = parent[node]
        path.reverse()
        return path if path and path[0] == start else []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(a: Coord, b: Coord) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _steer(src: Coord, dst: Coord, step: float, H: int, W: int) -> Coord:
    """Move from src toward dst by at most step cells."""
    d = _dist(src, dst)
    if d <= step:
        return dst
    ratio = step / d
    r = max(0, min(H - 1, int(round(src[0] + ratio * (dst[0] - src[0])))))
    c = max(0, min(W - 1, int(round(src[1] + ratio * (dst[1] - src[1])))))
    return (r, c)


def _los(p1: Coord, p2: Coord, grid: np.ndarray) -> bool:
    """Bresenham line-of-sight check between two grid cells."""
    r1, c1 = p1
    r2, c2 = p2
    H, W = grid.shape
    dr, dc = abs(r2 - r1), abs(c2 - c1)
    sr, sc = (1 if r2 > r1 else -1), (1 if c2 > c1 else -1)
    err = dr - dc
    r, c = r1, c1
    while True:
        if not (0 <= r < H and 0 <= c < W) or grid[r, c] == 1:
            return False
        if r == r2 and c == c2:
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
