"""Prompt templates for the Generator agent."""

from __future__ import annotations

from agenticpathplanner.core.types import MapInstance

SYSTEM_PROMPT = """\
You are an expert algorithm designer for autonomous path planning systems.

Given a mission description and map details, write a Python function
`plan(grid, start, goal)` that implements a path planning algorithm
specifically tailored to the mission's constraints and priorities.

Requirements:
  - Function signature: plan(grid, start, goal) -> list[tuple[int,int]]
  - grid: numpy uint8 array (0=free, 1=obstacle)
  - start, goal: (row, col) tuples
  - Returns a collision-free path or [] if none found
  - 8-connected movement
  - Allowed imports: numpy, math, heapq, collections

The algorithm must reflect the mission priorities — not just find any path,
but the RIGHT path for this mission (e.g. maximise clearance for a rescue
mission, or minimise length for a time-critical delivery).

Output ONLY the Python function code. No markdown, no explanation.
"""


class GeneratorPromptBuilder:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build(self, instance: MapInstance, mission: str) -> str:
        grid_section = ""
        if instance.height * instance.width <= 900:
            grid_section = f"\nGrid (S=start, G=goal, .=free, #=obstacle):\n{instance.as_text_grid()}\n"

        return (
            f"Map: {instance.map_id}  ({instance.height}×{instance.width}, {instance.difficulty})\n"
            f"Start: {instance.start}   Goal: {instance.goal}\n"
            f"Obstacle ratio: {instance.grid.mean():.2%}\n"
            f"{grid_section}"
            f"\nMission description:\n{mission}\n"
            f"\nDesign plan(grid, start, goal) that best serves this mission. "
            f"Output ONLY the function code."
        )
