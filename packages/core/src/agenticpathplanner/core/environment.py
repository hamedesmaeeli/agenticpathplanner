"""2D binary occupancy grid — runtime helper used by all planners."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from agenticpathplanner.core.types import Coord, MapInstance

_MOVES_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_MOVES_4 = [(-1,0),(0,-1),(0,1),(1,0)]


class GridWorld:
    """Wraps a MapInstance with runtime helpers for neighbour queries and rendering."""

    def __init__(self, instance: MapInstance, connectivity: int = 8):
        assert connectivity in (4, 8)
        self.instance = instance
        self.grid = instance.grid
        self.start = instance.start
        self.goal = instance.goal
        self.height, self.width = self.grid.shape
        self._moves = _MOVES_8 if connectivity == 8 else _MOVES_4

    def is_free(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0

    def neighbours(self, r: int, c: int) -> Iterator[Coord]:
        for dr, dc in self._moves:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                yield (nr, nc)

    def step_cost(self, src: Coord, dst: Coord) -> float:
        dr, dc = dst[0] - src[0], dst[1] - src[1]
        return float(np.sqrt(dr * dr + dc * dc))

    def heuristic(self, pos: Coord, goal: Coord | None = None) -> float:
        g = goal if goal is not None else self.goal
        return float(np.sqrt((pos[0] - g[0]) ** 2 + (pos[1] - g[1]) ** 2))

    def render(
        self,
        path=None,
        title: str = "",
        save_path: str | None = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; cannot render.")
            return

        display = np.ones((*self.grid.shape, 3), dtype=float)
        display[self.grid == 1] = [0.1, 0.1, 0.1]
        if path:
            for r, c in path:
                display[r, c] = [0.2, 0.6, 1.0]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(display, origin="upper")
        sr, sc = self.start
        gr, gc = self.goal
        ax.plot(sc, sr, "gs", markersize=10, label="Start")
        ax.plot(gc, gr, "r*", markersize=12, label="Goal")
        if title:
            ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
