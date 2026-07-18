"""Abstract planner interface — all planners across all subprojects implement this."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agenticpathplanner.core.types import MapInstance, Path


@dataclass
class PlanResult:
    path: Path
    elapsed_s: float
    success: bool
    metadata: dict = field(default_factory=dict)


class BasePlanner(ABC):
    """Every planner (LLM, classical, hybrid…) implements this interface."""

    name: str = "base"

    def solve(self, instance: MapInstance) -> PlanResult:
        t0 = time.perf_counter()
        path = self._plan(instance)
        elapsed = time.perf_counter() - t0
        success = bool(path and path[0] == instance.start and path[-1] == instance.goal)
        return PlanResult(path=path, elapsed_s=elapsed, success=success)

    @abstractmethod
    def _plan(self, instance: MapInstance) -> Path:
        """Return an ordered list of (row, col) waypoints, or [] on failure."""
