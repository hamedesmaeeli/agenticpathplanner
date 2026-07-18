"""CEC2024 standard path quality metrics: PLR, SD, SM, SR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from agenticpathplanner.core.types import Coord, Path


@dataclass
class PathMetrics:
    plr: float
    sd: float
    sm: float
    valid: bool
    path_length: float

    def as_dict(self) -> dict:
        return {"plr": round(self.plr, 4), "sd": round(self.sd, 4),
                "sm": round(self.sm, 4), "valid": self.valid,
                "path_length": round(self.path_length, 4)}


def evaluate_path(path: Path, grid: np.ndarray, start: Coord, goal: Coord) -> PathMetrics:
    if not path:
        return PathMetrics(float("inf"), 0.0, float("inf"), False, 0.0)
    valid = _check_validity(path, grid, start, goal)
    length = _length(path)
    straight = _straight(start, goal)
    plr = length / straight if straight > 0 else float("inf")
    return PathMetrics(plr=plr, sd=_safety(path, grid), sm=_smoothness(path),
                       valid=valid, path_length=length)


def aggregate_runs(metrics_list: list[PathMetrics]) -> dict:
    if not metrics_list:
        return {}
    valid = [m for m in metrics_list if m.valid]
    sr = len(valid) / len(metrics_list)
    if not valid:
        return {"sr": 0.0, "plr_mean": None, "sd_mean": None, "sm_mean": None}
    return {
        "sr": round(sr, 4),
        "plr_mean": round(float(np.mean([m.plr for m in valid])), 4),
        "plr_std":  round(float(np.std([m.plr for m in valid])), 4),
        "sd_mean":  round(float(np.mean([m.sd for m in valid])), 4),
        "sm_mean":  round(float(np.mean([m.sm for m in valid])), 4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_validity(path, grid, start, goal) -> bool:
    if not path or path[0] != start or path[-1] != goal:
        return False
    H, W = grid.shape
    for r, c in path:
        if not (0 <= r < H and 0 <= c < W) or grid[r, c] == 1:
            return False
    for i in range(len(path) - 1):
        if not _segment_free(path[i], path[i + 1], grid):
            return False
    return True


def _segment_free(p1, p2, grid) -> bool:
    r1, c1 = p1; r2, c2 = p2
    H, W = grid.shape
    dr, dc = abs(r2-r1), abs(c2-c1)
    sr, sc = (1 if r2>r1 else -1), (1 if c2>c1 else -1)
    err = dr - dc
    r, c = r1, c1
    while True:
        if not (0<=r<H and 0<=c<W) or grid[r, c] == 1:
            return False
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc
    return True


def _length(path) -> float:
    return sum(math.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2)
               for i in range(len(path)-1))


def _straight(start, goal) -> float:
    return math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)


def _safety(path, grid) -> float:
    obs = np.argwhere(grid == 1).astype(float)
    if obs.size == 0:
        return float("inf")
    return min(float(np.linalg.norm(obs - np.array([r, c], float), axis=1).min())
               for r, c in path)


def _smoothness(path) -> float:
    if len(path) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(path)-1):
        v1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        v2 = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        n1 = math.sqrt(v1[0]**2+v1[1]**2); n2 = math.sqrt(v2[0]**2+v2[1]**2)
        if n1 and n2:
            cos_a = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
            total += math.acos(cos_a)
    return total
