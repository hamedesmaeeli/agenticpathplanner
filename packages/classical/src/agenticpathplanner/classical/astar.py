"""A* planner — 8-connected, Euclidean heuristic, optimal path length."""

from __future__ import annotations

import heapq
import math
from typing import Tuple

from agenticpathplanner.core.planner import BasePlanner, PlanResult
from agenticpathplanner.core.types import Coord, MapInstance, Path

_MOVES = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


class AStarPlanner(BasePlanner):
    """Classical A* — used as the deterministic optimal baseline."""

    name = "astar"

    def _plan(self, instance: MapInstance) -> Path:
        grid, start, goal = instance.grid, instance.start, instance.goal
        H, W = grid.shape

        h = lambda p: math.sqrt((p[0]-goal[0])**2 + (p[1]-goal[1])**2)

        heap: list[tuple[float, Coord]] = [(h(start), start)]
        came_from: dict[Coord, Coord] = {}
        g: dict[Coord, float] = {start: 0.0}

        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                return _reconstruct(came_from, cur)
            for dr, dc in _MOVES:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == 1:
                    continue
                nb: Coord = (nr, nc)
                tg = g[cur] + math.sqrt(dr*dr + dc*dc)
                if tg < g.get(nb, float("inf")):
                    came_from[nb] = cur
                    g[nb] = tg
                    heapq.heappush(heap, (tg + h(nb), nb))

        return []


def _reconstruct(came_from: dict, current: Coord) -> Path:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
