"""GeneratorAgent — Idea 2: LLM as Mission-Specific Algorithm Generator.

Cycle:
  perceive  → read mission description + map from AgentContext
  reason    → ask LLM to write a plan(grid, start, goal) function for this mission
  act       → execute generated code in a sandbox; return the path
  reflect   → log the generated code for reproducibility
"""

from __future__ import annotations

import logging

from agenticpathplanner.core.agent import AgentContext, AgentResult, BaseAgent
from agenticpathplanner.generator.executor import CodeExecutor
from agenticpathplanner.generator.prompts import GeneratorPromptBuilder
from agenticpathplanner.llm.client import GenerationConfig, OllamaClient

log = logging.getLogger(__name__)


class GeneratorAgent(BaseAgent):
    """Generates a mission-tailored planning algorithm on-the-fly using an LLM.

    The `context.mission` field should describe the mission in natural language,
    e.g. "Search and rescue in collapsed building. Prioritise clearance from
    unstable walls. Accept longer paths to stay safe."
    """

    name = "generator"

    def __init__(self, model: str = "llama3.2", temperature: float = 0.3,
                 ollama_host: str | None = None):
        self.client = OllamaClient(GenerationConfig(model=model, temperature=temperature),
                                   host=ollama_host)
        self.prompt_builder = GeneratorPromptBuilder()
        self.executor = CodeExecutor()

    # ------------------------------------------------------------------
    # Agentic cycle
    # ------------------------------------------------------------------

    def reason(self, observation: dict, context: AgentContext) -> dict:
        mission = context.mission or "Find the shortest collision-free path."
        prompt = self.prompt_builder.build(context.instance, mission)
        try:
            response = self.client.generate(
                prompt=prompt, system=self.prompt_builder.system_prompt()
            )
            return {"code": response.content, "model": response.model, "mission": mission}
        except RuntimeError as exc:
            log.error("Generator LLM call failed: %s", exc)
            return {"code": "", "error": str(exc), "mission": mission}

    def act(self, plan: dict, context: AgentContext) -> AgentResult:
        code = plan.get("code", "")
        path, stored_code = self.executor.execute(code, context.instance)

        return AgentResult(
            path=path,
            success=bool(path and path[0] == context.instance.start
                         and path[-1] == context.instance.goal),
            agent_name=self.name,
            reasoning=f"Generated algorithm for mission: {plan.get('mission','')}",
            generated_code=stored_code,
            metadata={"model": plan.get("model", ""), "mission": plan.get("mission", "")},
        )

    def reflect(self, result: AgentResult, context: AgentContext) -> AgentResult:
        if result.generated_code:
            log.info("Generated code (%d chars) for map %s",
                     len(result.generated_code), context.instance.map_id)
        return result
