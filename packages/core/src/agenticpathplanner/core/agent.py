"""Base agentic AI agent interface.

Every path planning agent (Refiner, Generator, Selector, …) implements
BaseAgent.  The four-step loop — perceive → reason → act → reflect — is
the standard agentic cycle and maps naturally to each subproject:

  Refiner   perceive existing path + env context → reason with LLM → act to refine
  Generator perceive mission description         → reason with LLM → act to generate code
  Selector  perceive real-time conditions        → reason with LLM → act to pick algorithm
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agenticpathplanner.core.types import MapInstance, Path


@dataclass
class AgentContext:
    """Everything an agent needs to know before acting."""

    instance: MapInstance
    environment: dict[str, Any] = field(default_factory=dict)
    mission: str = ""
    initial_path: Path = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_observation(self) -> dict:
        return {
            "map_id": self.instance.map_id,
            "difficulty": self.instance.difficulty,
            "size": f"{self.instance.height}×{self.instance.width}",
            "start": self.instance.start,
            "goal": self.instance.goal,
            "mission": self.mission,
            "environment": self.environment,
            "initial_path_length": len(self.initial_path),
        }


@dataclass
class AgentResult:
    """What every agent returns after completing its cycle."""

    path: Path
    success: bool
    agent_name: str
    reasoning: str = ""
    selected_algorithm: str | None = None
    generated_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Agentic AI agent interface for path planning subprojects.

    Subclass this and implement `reason` and `act`.
    Override `perceive` or `reflect` for richer agent behaviour.
    """

    name: str = "base_agent"

    def run(self, context: AgentContext) -> AgentResult:
        """Full agentic cycle: perceive → reason → act → reflect."""
        observation = self.perceive(context)
        plan = self.reason(observation, context)
        result = self.act(plan, context)
        return self.reflect(result, context)

    def perceive(self, context: AgentContext) -> dict:
        """Convert context to a structured observation dict."""
        return context.as_observation()

    @abstractmethod
    def reason(self, observation: dict, context: AgentContext) -> dict:
        """Use LLM or logic to decide what to do. Returns an intermediate plan dict."""

    @abstractmethod
    def act(self, plan: dict, context: AgentContext) -> AgentResult:
        """Execute the plan and return a result."""

    def reflect(self, result: AgentResult, context: AgentContext) -> AgentResult:
        """Optional post-processing or quality check. Default: pass-through."""
        return result
