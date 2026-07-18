"""Tests for agenticpathplanner.llm (mocked Ollama — no live server)."""

from unittest.mock import MagicMock

import pytest

from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
from agenticpathplanner.llm.client import LLMResponse
from agenticpathplanner.llm.planner import LLMPlanner, _coerce
from agenticpathplanner.llm.prompts import PromptBuilder


class TestPromptBuilder:
    def test_path_prompt_contains_start_goal(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        pb = PromptBuilder(mode="generate_path")
        assert str(inst.start) in pb.build(inst)
        assert str(inst.goal) in pb.build(inst)

    def test_algorithm_system_has_plan_signature(self):
        assert "plan(grid, start, goal)" in PromptBuilder(mode="generate_algorithm").system_prompt()

    def test_modes_have_different_system_prompts(self):
        assert PromptBuilder("generate_path").system_prompt() != PromptBuilder("generate_algorithm").system_prompt()


class TestCoerce:
    def test_list_of_lists(self):
        assert _coerce([[0,0],[1,1]]) == [(0,0),(1,1)]

    def test_empty(self):
        assert _coerce([]) == []


def _make_planner(mode="generate_path") -> LLMPlanner:
    p = LLMPlanner.__new__(LLMPlanner)
    p.mode = mode
    p.name = f"llm_{mode}"
    p.prompt_builder = PromptBuilder(mode=mode)
    p.client = MagicMock()
    return p


class TestLLMPlannerMocked:
    def test_generate_path_success(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        p = _make_planner("generate_path")
        p.client.generate.return_value = LLMResponse("[(1,1),(2,2),(8,8)]", "mock")
        result = p.solve(inst)
        assert result.path == [(1,1),(2,2),(8,8)]

    def test_generate_algorithm_executes(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        p = _make_planner("generate_algorithm")
        p.client.generate.return_value = LLMResponse(
            "def plan(grid, start, goal):\n    return [start, goal]", "mock"
        )
        result = p.solve(inst)
        assert result.path[0] == inst.start
        assert result.path[-1] == inst.goal

    def test_ollama_error_gives_empty_path(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        p = _make_planner()
        p.client.generate.side_effect = RuntimeError("offline")
        result = p.solve(inst)
        assert not result.success

    def test_markdown_fence_stripped(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        p = _make_planner("generate_path")
        p.client.generate.return_value = LLMResponse("```python\n[(1,1),(8,8)]\n```", "mock")
        result = p.solve(inst)
        assert result.path == [(1,1),(8,8)]
