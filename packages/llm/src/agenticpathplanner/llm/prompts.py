"""Prompt templates for generate_path and generate_algorithm modes."""

from __future__ import annotations

from agenticpathplanner.core.types import MapInstance

SYSTEM_GENERATE_PATH = """\
You are an expert path planning algorithm designer solving 2D grid path planning problems.

Grid conventions: (row, col) with (0,0) top-left. 0=free, 1=obstacle. 8-connected movement.
The path must be collision-free from S (start) to G (goal).

Output ONLY a Python list of (row, col) tuples. No explanation, no markdown, just the list.
Example: [(0, 0), (1, 1), (2, 2)]
"""

SYSTEM_GENERATE_ALGORITHM = """\
You are an expert algorithm designer specialising in 2D path planning.

Write a Python function `plan(grid, start, goal)` that:
  - Takes: grid (numpy uint8 array, 0=free, 1=obstacle), start (row,col), goal (row,col)
  - Returns: list of (row,col) tuples from start to goal, or [] if no path found
  - Uses 8-connected movement
  - Imports allowed: numpy, math, heapq, collections only

The algorithm must be tailored to the specific problem instance described below.
Output ONLY the Python function code. No markdown, no explanation.
"""


class PromptBuilder:
    def __init__(self, mode: str = "generate_path", include_grid: bool = True):
        assert mode in ("generate_path", "generate_algorithm")
        self.mode = mode
        self.include_grid = include_grid

    def system_prompt(self) -> str:
        return SYSTEM_GENERATE_PATH if self.mode == "generate_path" else SYSTEM_GENERATE_ALGORITHM

    def build(self, instance: MapInstance) -> str:
        header = (
            f"Map ID: {instance.map_id}  ({instance.height}×{instance.width})\n"
            f"Difficulty: {instance.difficulty}\n"
            f"Start: {instance.start}\n"
            f"Goal:  {instance.goal}\n"
        )
        grid_section = ""
        if self.include_grid and instance.height * instance.width <= 900:
            grid_section = f"\nGrid (S=start, G=goal, .=free, #=obstacle):\n{instance.as_text_grid()}\n"
        else:
            grid_section = f"\nObstacle ratio: {instance.grid.mean():.2%}\n"

        if self.mode == "generate_path":
            tail = "\nReturn a Python list of (row, col) tuples. Output ONLY the list."
        else:
            tail = (
                "\nDesign plan(grid, start, goal) tailored to this map's characteristics. "
                "Output ONLY the Python function."
            )
        return header + grid_section + tail
