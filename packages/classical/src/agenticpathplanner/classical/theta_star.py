"""Theta* — any-angle path planner on 8-connected grids.

Extends A* with a line-of-sight (LoS) check: when expanding a node n,
if the grandparent g has direct LoS to n's neighbour n', connect g→n'
directly, bypassing n. This produces shorter, any-angle paths without
being restricted to grid-edge directions.

Reference: Nash & Koenig (2007). Any-angle path planning. AAAI 2007.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import Coord, MapInstance, Path

_MOVES = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


class ThetaStarPlanner(BasePlanner):
    """Any-angle Theta* — produces shorter, smoother paths than A* on grids."""

    name = "theta_star"

    def _plan(self, instance: MapInstance) -> Path:
        grid  = instance.grid
        start = instance.start
        goal  = instance.goal
        H, W  = grid.shape

        def h(p: Coord) -> float:
            return math.sqrt((p[0]-goal[0])**2 + (p[1]-goal[1])**2)

        open_heap: list[tuple[float, Coord]] = []
        heapq.heappush(open_heap, (h(start), start))

        came_from: dict[Coord, Coord] = {start: start}
        g: dict[Coord, float]         = {start: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)

            if cur == goal:
                return _reconstruct(came_from, goal, start)

            parent = came_from.get(cur, cur)

            for dr, dc in _MOVES:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == 1:
                    continue
                nb: Coord = (nr, nc)

                # Theta*: try to connect grandparent → neighbour directly
                if _los(parent, nb, grid):
                    pr, pc = parent
                    tg = g[parent] + math.sqrt((nr-pr)**2 + (nc-pc)**2)
                    if tg < g.get(nb, float("inf")):
                        came_from[nb] = parent
                        g[nb] = tg
                        heapq.heappush(open_heap, (tg + h(nb), nb))
                else:
                    tg = g[cur] + math.sqrt(dr*dr + dc*dc)
                    if tg < g.get(nb, float("inf")):
                        came_from[nb] = cur
                        g[nb] = tg
                        heapq.heappush(open_heap, (tg + h(nb), nb))

        return []


def _los(p1: Coord, p2: Coord, grid: np.ndarray) -> bool:
    """Bresenham line-of-sight check between two grid cells."""
    r1, c1 = p1
    r2, c2 = p2
    H, W = grid.shape
    dr, dc = abs(r2-r1), abs(c2-c1)
    sr, sc = (1 if r2>r1 else -1), (1 if c2>c1 else -1)
    err = dr - dc
    r, c = r1, c1
    while True:
        if not (0 <= r < H and 0 <= c < W) or grid[r, c] == 1:
            return False
        if r == r2 and c == c2:
            return True
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc


def _reconstruct(came_from: dict[Coord, Coord], goal: Coord, start: Coord) -> Path:
    path = [goal]
    cur  = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path
