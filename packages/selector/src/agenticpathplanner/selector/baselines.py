"""Non-LLM selector baselines for comparison with SelectorAgent.

Three baselines bracket the LLM Selector's performance:

  RandomSelector    — uniform random algorithm choice (floor baseline)
  RuleBasedSelector — expert hand-coded condition→algorithm mapping
  OracleSelector    — post-hoc optimal choice per map (ceiling baseline)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import MapInstance, Path

if TYPE_CHECKING:
    from agenticpathplanner.selector.registry import AlgorithmRegistry


class RandomSelector(BasePlanner):
    """Selects a uniformly random algorithm from the registry.

    Establishes the performance floor: if the LLM Selector barely beats
    random, the LLM contribution is weak.
    """

    name = "random_selector"

    def __init__(self, registry: "AlgorithmRegistry", seed: int | None = None):
        self.registry = registry
        self._rng = random.Random(seed)

    def _plan(self, instance: MapInstance) -> Path:
        names = self.registry.names()
        if not names:
            return []
        chosen = self._rng.choice(names)
        planner = self.registry.get(chosen)
        if planner is None:
            return []
        return planner.solve(instance).path


class RuleBasedSelector(BasePlanner):
    """Selects an algorithm using hand-coded obstacle-density rules.

    Rules (mirroring common expert intuition):
      obstacle_ratio < 0.15  →  astar      (open grid, A* is optimal)
      0.15 ≤ ratio < 0.30    →  theta_star (moderate clutter, smoother paths)
      ratio ≥ 0.30           →  apf_refined (dense clutter, clearance priority)

    Acts as the 'human expert' comparison for the LLM Selector.
    """

    name = "rule_based_selector"

    _RULES = [
        (0.15, "astar"),
        (0.30, "theta_star"),
        (1.00, "apf_refined"),
    ]

    def __init__(self, registry: "AlgorithmRegistry"):
        self.registry = registry

    def _plan(self, instance: MapInstance) -> Path:
        ratio = float(instance.grid.mean())
        chosen = "astar"
        for threshold, algo in self._RULES:
            if ratio < threshold:
                chosen = algo
                break

        planner = self.registry.get(chosen)
        if planner is None:
            planner = self.registry.get("astar")
        if planner is None:
            return []
        return planner.solve(instance).path

    def selection_reason(self, instance: MapInstance) -> str:
        ratio = float(instance.grid.mean())
        for threshold, algo in self._RULES:
            if ratio < threshold:
                return f"obstacle_ratio={ratio:.2%} < {threshold:.0%} → {algo}"
        return f"obstacle_ratio={ratio:.2%} ≥ 0.30 → apf_refined"


class OracleSelector(BasePlanner):
    """Selects the known-best algorithm per map (post-hoc ceiling baseline).

    The oracle_map is populated after a first run by identifying which
    algorithm achieved the lowest PLR on each map. It represents the
    theoretical ceiling for any selection policy given the same registry.

    Usage:
        oracle = OracleSelector(registry, oracle_map={
            "cec2024_s01": "astar",
            "cec2024_m01": "theta_star",
            "cec2024_c01": "apf_refined",
        })
    """

    name = "oracle_selector"

    def __init__(
        self,
        registry: "AlgorithmRegistry",
        oracle_map: dict[str, str] | None = None,
        default_algo: str = "astar",
    ):
        self.registry     = registry
        self.oracle_map   = oracle_map or {}
        self.default_algo = default_algo

    def _plan(self, instance: MapInstance) -> Path:
        chosen  = self.oracle_map.get(instance.map_id, self.default_algo)
        planner = self.registry.get(chosen) or self.registry.get(self.default_algo)
        if planner is None:
            return []
        return planner.solve(instance).path

    def set_oracle(self, map_id: str, algo: str) -> None:
        self.oracle_map[map_id] = algo
