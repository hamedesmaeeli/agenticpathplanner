"""Fundamental types shared by all agenticpathplanner subprojects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

Coord = Tuple[int, int]          # (row, col) in grid coordinates
Path  = List[Coord]              # ordered list of grid coordinates


@dataclass
class MapInstance:
    """A single path planning problem instance.

    Used by all subprojects (benchmarks, planners, evaluators) as the
    common currency for passing maps between components.
    """

    map_id: str
    grid: np.ndarray              # shape (H, W), dtype uint8; 0=free, 1=obstacle
    start: Coord                  # (row, col)
    goal: Coord                   # (row, col)
    difficulty: str = "medium"    # "simple" | "medium" | "complex"
    description: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def is_free(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.grid[row, col] == 0

    def straight_line_distance(self) -> float:
        sr, sc = self.start
        gr, gc = self.goal
        return float(np.sqrt((gr - sr) ** 2 + (gc - sc) ** 2))

    def as_text_grid(self, path: Path | None = None) -> str:
        """ASCII representation for LLM prompts and debugging."""
        symbols = {0: ".", 1: "#"}
        path_set = set(path) if path else set()
        rows = []
        for r in range(self.height):
            chars = []
            for c in range(self.width):
                if (r, c) == self.start:
                    chars.append("S")
                elif (r, c) == self.goal:
                    chars.append("G")
                elif (r, c) in path_set:
                    chars.append("*")
                else:
                    chars.append(symbols[self.grid[r, c]])
            rows.append(" ".join(chars))
        return "\n".join(rows)
