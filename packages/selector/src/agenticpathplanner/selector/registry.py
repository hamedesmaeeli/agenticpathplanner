"""Algorithm registry — maps string names to BasePlanner instances.

New planners from any subproject can be registered here so the Selector
agent can discover and invoke them.
"""

from __future__ import annotations

from agenticpathplanner.core.planner import BasePlanner


class AlgorithmRegistry:
    """A named registry of path planning algorithms."""

    def __init__(self):
        self._registry: dict[str, BasePlanner] = {}

    def register(self, name: str, planner: BasePlanner) -> None:
        self._registry[name] = planner

    def get(self, name: str) -> BasePlanner | None:
        return self._registry.get(name)

    def names(self) -> list[str]:
        return list(self._registry.keys())

    def descriptions(self) -> dict[str, str]:
        """Return name → docstring summary for use in LLM prompts."""
        return {
            name: (type(p).__doc__ or "").strip().splitlines()[0]
            for name, p in self._registry.items()
        }

    @classmethod
    def default(cls) -> "AlgorithmRegistry":
        """Build a registry with all built-in algorithms pre-registered."""
        from agenticpathplanner.classical.astar import AStarPlanner
        from agenticpathplanner.classical.theta_star import ThetaStarPlanner
        from agenticpathplanner.classical.apf import APFRefiner
        from agenticpathplanner.classical.rrt_star import RRTStarPlanner
        from agenticpathplanner.classical.mpc import MPCPlanner
        reg = cls()
        reg.register("astar",       AStarPlanner())
        reg.register("theta_star",  ThetaStarPlanner())
        reg.register("apf_refined", APFRefiner())
        reg.register("rrt_star",    RRTStarPlanner())
        reg.register("mpc",         MPCPlanner())
        return reg
