"""Prompt templates for the Refiner agent."""

from __future__ import annotations

from agenticpathplanner.core.types import MapInstance, Path

SYSTEM_PROMPT = """\
You are an expert UAV path planning advisor. A classical algorithm has already
produced a geometrically valid path. Your task is to refine it using the
environmental context provided (obstacles, wind, interference zones, safety
margins, etc.) to make the path safer and more operationally sound.

Rules:
  - Output ONLY a Python list of (row, col) tuples — no explanation, no markdown.
  - The refined path must still start at S and end at G.
  - You may add, remove, or shift waypoints to improve safety.
  - Prefer gradual changes; avoid drastic deviations unless the context demands it.
Example output: [(0,0), (1,1), (2,2), (5,5)]
"""


class RefinerPromptBuilder:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build(
        self,
        instance: MapInstance,
        initial_path: Path,
        environment: dict,
    ) -> str:
        env_lines = "\n".join(f"  - {k}: {v}" for k, v in environment.items()) or "  (none)"
        path_str = str(initial_path[:20]) + (" ..." if len(initial_path) > 20 else "")
        grid_section = ""
        if instance.height * instance.width <= 900:
            grid_section = f"\nGrid (S=start, G=goal, .=free, #=obstacle):\n{instance.as_text_grid(initial_path)}\n"

        return (
            f"Map: {instance.map_id}  ({instance.height}×{instance.width}, {instance.difficulty})\n"
            f"Start: {instance.start}   Goal: {instance.goal}\n"
            f"\nInitial path ({len(initial_path)} waypoints):\n{path_str}\n"
            f"{grid_section}"
            f"\nEnvironmental context:\n{env_lines}\n"
            f"\nRefine the path to maximise safety given the context above. "
            f"Output ONLY the revised (row, col) list."
        )
