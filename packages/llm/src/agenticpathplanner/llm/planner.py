"""LLM-based path planner — generate_path and generate_algorithm modes."""

from __future__ import annotations

import ast
import logging
import re
import textwrap

import numpy as np

from agenticpathplanner.core.planner import BasePlanner
from agenticpathplanner.core.types import MapInstance, Path
from agenticpathplanner.llm.client import GenerationConfig, LLMResponse, OllamaClient
from agenticpathplanner.llm.prompts import PromptBuilder

log = logging.getLogger(__name__)


class LLMPlanner(BasePlanner):
    """Delegates path planning to a local Ollama language model."""

    name = "llm"

    def __init__(
        self,
        model: str = "llama3.2",
        mode: str = "generate_path",
        temperature: float = 0.2,
        include_grid: bool = True,
        ollama_host: str | None = None,
    ):
        self.mode = mode
        self.name = f"llm_{mode}"
        self.client = OllamaClient(GenerationConfig(model=model, temperature=temperature),
                                   host=ollama_host)
        self.prompt_builder = PromptBuilder(mode=mode, include_grid=include_grid)

    def _plan(self, instance: MapInstance) -> Path:
        try:
            response = self.client.generate(
                prompt=self.prompt_builder.build(instance),
                system=self.prompt_builder.system_prompt(),
            )
        except RuntimeError as exc:
            log.error("LLM call failed: %s", exc)
            return []
        return (self._parse_path(response, instance)
                if self.mode == "generate_path"
                else self._execute_algorithm(response, instance))

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_path(self, response: LLMResponse, instance: MapInstance) -> Path:
        text = re.sub(r"```[a-z]*\n?", "", response.content).strip("`").strip()
        try:
            result = ast.literal_eval(text)
            path = _coerce(result)
            if path:
                return path
        except (ValueError, SyntaxError):
            pass
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if pairs:
            return [(int(r), int(c)) for r, c in pairs]
        log.warning("Could not parse path for map %s", instance.map_id)
        return []

    def _execute_algorithm(self, response: LLMResponse, instance: MapInstance) -> Path:
        code = re.sub(r"```[a-z]*\n?", "", response.content).strip("`").strip()
        if not code:
            return []
        import math as _math, heapq as _heapq, collections as _collections
        ns: dict = {
            "numpy": np, "np": np,
            "math": _math, "heapq": _heapq, "collections": _collections,
            "__builtins__": {k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
                             if k in ("range","len","list","tuple","int","float","bool",
                                      "min","max","abs","sum","sorted","enumerate","zip",
                                      "print","isinstance","type","hasattr","__import__")},
        }
        try:
            exec(textwrap.dedent(code), ns)
        except Exception as exc:
            log.warning("Algorithm compile error for %s: %s", instance.map_id, exc)
            return []
        plan_fn = ns.get("plan")
        if not callable(plan_fn):
            log.warning("LLM code has no callable `plan` function")
            return []
        try:
            return _coerce(plan_fn(instance.grid, instance.start, instance.goal))
        except Exception as exc:
            log.warning("Algorithm runtime error for %s: %s", instance.map_id, exc)
            return []


def _coerce(obj) -> Path:
    if not obj:
        return []
    return [(int(x[0]), int(x[1])) for x in obj if isinstance(x, (list, tuple)) and len(x) == 2]
