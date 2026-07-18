"""RefinerAgent — Idea 1: LLM as path Refiner.

Cycle:
  perceive  → extract initial path + env context from AgentContext
  reason    → ask LLM how to improve the path given the context
  act       → parse the LLM-suggested waypoints; fall back to initial path on failure
  reflect   → validate the refined path is still collision-free
"""

from __future__ import annotations

import ast
import logging
import re

from agenticpathplanner.core.agent import AgentContext, AgentResult, BaseAgent
from agenticpathplanner.core.environment import GridWorld
from agenticpathplanner.core.types import Path
from agenticpathplanner.llm.client import GenerationConfig, OllamaClient
from agenticpathplanner.refiner.prompts import RefinerPromptBuilder

log = logging.getLogger(__name__)


class RefinerAgent(BaseAgent):
    """Uses an LLM to semantically refine a path produced by a classical planner.

    Expects `context.initial_path` to be populated (e.g. by A* beforehand).
    `context.environment` carries domain context such as wind zones,
    interference radii, and no-fly areas.
    """

    name = "refiner"

    def __init__(self, model: str = "llama3.2", temperature: float = 0.2,
                 ollama_host: str | None = None):
        self.client = OllamaClient(GenerationConfig(model=model, temperature=temperature),
                                   host=ollama_host)
        self.prompt_builder = RefinerPromptBuilder()

    # ------------------------------------------------------------------
    # Agentic cycle
    # ------------------------------------------------------------------

    def reason(self, observation: dict, context: AgentContext) -> dict:
        prompt = self.prompt_builder.build(
            context.instance, context.initial_path, context.environment
        )
        try:
            response = self.client.generate(
                prompt=prompt, system=self.prompt_builder.system_prompt()
            )
            return {"raw": response.content, "model": response.model}
        except RuntimeError as exc:
            log.error("Refiner LLM call failed: %s", exc)
            return {"raw": "", "error": str(exc)}

    def act(self, plan: dict, context: AgentContext) -> AgentResult:
        raw = plan.get("raw", "")
        path = self._parse(raw) if raw else []

        if not path:
            log.warning("Refiner could not parse LLM response; keeping initial path.")
            path = context.initial_path

        return AgentResult(
            path=path,
            success=bool(path),
            agent_name=self.name,
            reasoning=raw[:300] if raw else "LLM call failed",
            metadata={"model": plan.get("model", ""), "fallback": not bool(self._parse(raw))},
        )

    def reflect(self, result: AgentResult, context: AgentContext) -> AgentResult:
        """Validate that the refined path is still collision-free."""
        if not result.path:
            return result
        world = GridWorld(context.instance)
        for r, c in result.path:
            if not world.is_free(r, c):
                log.warning("Refined path hits obstacle at (%d,%d); reverting to initial.", r, c)
                result.path = context.initial_path
                result.metadata["reverted"] = True
                break
        return result

    # ------------------------------------------------------------------

    def _parse(self, text: str) -> Path:
        text = re.sub(r"```[a-z]*\n?", "", text).strip("`").strip()
        try:
            result = ast.literal_eval(text)
            if isinstance(result, list):
                return [(int(x[0]), int(x[1])) for x in result
                        if isinstance(x, (list, tuple)) and len(x) == 2]
        except (ValueError, SyntaxError):
            pass
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        return [(int(r), int(c)) for r, c in pairs]
