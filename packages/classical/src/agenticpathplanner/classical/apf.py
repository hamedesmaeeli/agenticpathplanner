"""APF post-processor — Artificial Potential Field path refiner.

Runs A* (or any supplied base planner) to get an initial valid path, then
applies repulsive forces from nearby obstacles to shift interior waypoints
away from obstacles, improving safety distance without replanning from scratch.

Start and goal waypoints are kept fixed. Each interior waypoint is iteratively
nudged by the sum of repulsive vectors from obstacles within `repulsion_radius`
cells, then snapped back to the nearest free integer cell.

Reference: Khatib (1986). Real-time obstacle avoidance. IJRR 5(1), 90–98.
"""

from __future__ import annotations

import math

import numpy as np

from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import Coord, MapInstance, Path


class APFRefiner(BasePlanner):
    """Refines an A* path by repelling waypoints from nearby obstacles."""

    name = "apf_refined"

    def __init__(
        self,
        base_planner: BasePlanner | None = None,
        iterations: int = 40,
        step: float = 0.35,
        repulsion_radius: float = 3.0,
        repulsion_gain: float = 1.5,
    ):
        self.base_planner     = base_planner or AStarPlanner()
        self.iterations       = iterations
        self.step             = step
        self.repulsion_radius = repulsion_radius
        self.repulsion_gain   = repulsion_gain

    def _plan(self, instance: MapInstance) -> Path:
        base_result = self.base_planner.solve(instance)
        if not base_result.success:
            return []
        refined = self._apply_apf(list(base_result.path), instance.grid)
        return refined

    # ------------------------------------------------------------------

    def _apply_apf(self, path: Path, grid: np.ndarray) -> Path:
        if len(path) <= 2:
            return path

        H, W = grid.shape
        obs   = np.argwhere(grid == 1).astype(float)

        # Work in float space for smooth nudging
        pts = [[float(r), float(c)] for r, c in path]

        for _ in range(self.iterations):
            for i in range(1, len(pts) - 1):  # keep start and goal fixed
                pr, pc = pts[i]
                fr, fc = 0.0, 0.0

                # Repulsion from each nearby obstacle
                for or_, oc in obs:
                    dr = pr - or_
                    dc = pc - oc
                    dist = math.sqrt(dr*dr + dc*dc)
                    if 0 < dist < self.repulsion_radius:
                        mag = self.repulsion_gain * (1.0/dist - 1.0/self.repulsion_radius) / (dist**2)
                        fr += mag * dr / dist
                        fc += mag * dc / dist

                # Clamp force magnitude
                force_mag = math.sqrt(fr*fr + fc*fc)
                if force_mag > 1.0:
                    fr /= force_mag
                    fc /= force_mag

                pts[i][0] += self.step * fr
                pts[i][1] += self.step * fc

                # Clamp to grid bounds
                pts[i][0] = max(0.0, min(H - 1.0, pts[i][0]))
                pts[i][1] = max(0.0, min(W - 1.0, pts[i][1]))

        # Snap to nearest free integer cell
        result: Path = [path[0]]
        for i in range(1, len(pts) - 1):
            snapped = _snap_to_free(pts[i][0], pts[i][1], grid, H, W)
            if snapped and snapped != result[-1]:
                result.append(snapped)
        result.append(path[-1])
        return result


def _snap_to_free(r: float, c: float, grid: np.ndarray, H: int, W: int) -> Coord | None:
    """Return nearest free integer cell to (r, c) via spiral search."""
    for radius in range(4):
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                nr, nc = int(round(r)) + dr, int(round(c)) + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                    return (nr, nc)
    return None
