"""Tests for agenticpathplanner.refiner (mocked LLM)."""

from unittest.mock import MagicMock

import pytest

from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
from agenticpathplanner.core.agent import AgentContext
from agenticpathplanner.llm.client import LLMResponse
from agenticpathplanner.refiner.agent import RefinerAgent
from agenticpathplanner.refiner.prompts import RefinerPromptBuilder


def _ctx(initial_path=None, env=None):
    inst = BUILTIN_MAPS["cec2024_s01"]
    return AgentContext(
        instance=inst,
        initial_path=initial_path or [inst.start, inst.goal],
        environment=env or {},
    )


class TestRefinerPrompt:
    def test_prompt_contains_initial_path(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        pb = RefinerPromptBuilder()
        prompt = pb.build(inst, [inst.start, inst.goal], {"wind": "15 knots N"})
        assert "wind" in prompt
        assert str(inst.start) in prompt

    def test_system_prompt_mentions_refine(self):
        assert "refine" in RefinerPromptBuilder().system_prompt().lower()


class TestRefinerAgent:
    def _agent_with_mock(self, response_text: str) -> RefinerAgent:
        agent = RefinerAgent.__new__(RefinerAgent)
        agent.name = "refiner"
        agent.prompt_builder = RefinerPromptBuilder()
        agent.client = MagicMock()
        agent.client.generate.return_value = LLMResponse(response_text, "mock")
        return agent

    def test_returns_refined_path(self):
        agent = self._agent_with_mock("[(1,1),(3,3),(8,8)]")
        result = agent.run(_ctx())
        assert result.path == [(1,1),(3,3),(8,8)]
        assert result.success

    def test_falls_back_to_initial_on_bad_response(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        initial = [inst.start, (4,4), inst.goal]
        agent = self._agent_with_mock("Sorry, I cannot help.")
        result = agent.run(_ctx(initial_path=initial))
        assert result.path == initial

    def test_reflect_reverts_obstacle_path(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        obstacle_path = [inst.start, (5, 2), inst.goal]  # (5,2) is an obstacle
        initial = [inst.start, inst.goal]
        agent = self._agent_with_mock(str(obstacle_path))
        result = agent.run(_ctx(initial_path=initial))
        assert (5, 2) not in result.path
