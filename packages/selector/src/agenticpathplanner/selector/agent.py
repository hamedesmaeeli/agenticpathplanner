"""SelectorAgent — Idea 3: LLM as Algorithm Selector (Meta-Path Planning).

Cycle:
  perceive  → read map conditions + environment from AgentContext
  reason    → ask LLM which registered algorithm best fits the conditions
  act       → execute the selected algorithm; fall back to default on failure
  reflect   → log the selection decision for auditability
"""

from __future__ import annotations

import logging
import re

from agenticpathplanner.core.agent import AgentContext, AgentResult, BaseAgent
from agenticpathplanner.llm.client import GenerationConfig, OllamaClient
from agenticpathplanner.selector.prompts import SelectorPromptBuilder
from agenticpathplanner.selector.registry import AlgorithmRegistry

log = logging.getLogger(__name__)

_DEFAULT_ALGORITHM = "astar"


class SelectorAgent(BaseAgent):
    """Uses an LLM to pick the most suitable algorithm from a registry.

    `context.environment` carries real-time conditions such as:
      {"obstacle_density": "high", "battery": "low", "target_moving": True}

    The LLM reasons over these conditions and the algorithm descriptions,
    then names the best algorithm. The agent executes it and returns the path.
    """

    name = "selector"

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.1,
        registry: AlgorithmRegistry | None = None,
        ollama_host: str | None = None,
    ):
        self.client = OllamaClient(GenerationConfig(model=model, temperature=temperature),
                                   host=ollama_host)
        self.prompt_builder = SelectorPromptBuilder()
        self.registry = registry or AlgorithmRegistry.default()

    # ------------------------------------------------------------------
    # Agentic cycle
    # ------------------------------------------------------------------

    def reason(self, observation: dict, context: AgentContext) -> dict:
        descriptions = self.registry.descriptions()
        conditions = {
            **context.environment,
            "map_difficulty": context.instance.difficulty,
            "obstacle_ratio": f"{context.instance.grid.mean():.2%}",
        }
        prompt = self.prompt_builder.build(descriptions, conditions)
        try:
            response = self.client.generate(
                prompt=prompt, system=self.prompt_builder.system_prompt()
            )
            selected, reasoning = self._parse_selection(response.content)
            return {"selected": selected, "reasoning": reasoning, "model": response.model}
        except RuntimeError as exc:
            log.error("Selector LLM call failed: %s", exc)
            return {"selected": _DEFAULT_ALGORITHM, "reasoning": "LLM unavailable; using default",
                    "error": str(exc)}

    def act(self, plan: dict, context: AgentContext) -> AgentResult:
        selected = plan.get("selected", _DEFAULT_ALGORITHM)
        reasoning = plan.get("reasoning", "")

        planner = self.registry.get(selected)
        if planner is None:
            log.warning("Selected algorithm '%s' not in registry; falling back to '%s'.",
                        selected, _DEFAULT_ALGORITHM)
            selected = _DEFAULT_ALGORITHM
            planner = self.registry.get(_DEFAULT_ALGORITHM)

        if planner is None:
            return AgentResult(path=[], success=False, agent_name=self.name,
                               reasoning="No planners available in registry.")

        result = planner.solve(context.instance)
        return AgentResult(
            path=result.path,
            success=result.success,
            agent_name=self.name,
            reasoning=reasoning,
            selected_algorithm=selected,
            metadata={"model": plan.get("model", ""), "elapsed_s": result.elapsed_s},
        )

    def reflect(self, result: AgentResult, context: AgentContext) -> AgentResult:
        log.info(
            "Selector chose '%s' for map '%s' (difficulty=%s): %s",
            result.selected_algorithm,
            context.instance.map_id,
            context.instance.difficulty,
            result.reasoning,
        )
        return result

    # ------------------------------------------------------------------

    def _parse_selection(self, text: str) -> tuple[str, str]:
        """Extract 'Selected: <name>' and 'Reasoning: <text>' from LLM output."""
        selected_match = re.search(r"Selected:\s*([^\n]+)", text, re.IGNORECASE)
        reasoning_match = re.search(r"Reasoning:\s*([^\n]+)", text, re.IGNORECASE)

        selected = selected_match.group(1).strip().lower() if selected_match else _DEFAULT_ALGORITHM
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()[:200]

        # Validate against registry
        if selected not in self.registry.names():
            log.warning("LLM selected unknown algorithm '%s'; falling back.", selected)
            selected = _DEFAULT_ALGORITHM

        return selected, reasoning
