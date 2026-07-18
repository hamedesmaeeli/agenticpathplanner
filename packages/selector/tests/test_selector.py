"""Tests for agenticpathplanner.selector (mocked LLM)."""

from unittest.mock import MagicMock

import pytest

from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.core.agent import AgentContext
from agenticpathplanner.llm.client import LLMResponse
from agenticpathplanner.selector.agent import SelectorAgent
from agenticpathplanner.selector.prompts import SelectorPromptBuilder
from agenticpathplanner.selector.registry import AlgorithmRegistry


def _ctx(env=None):
    return AgentContext(
        instance=BUILTIN_MAPS["cec2024_s01"],
        environment=env or {"obstacle_density": "low"},
    )


def _registry():
    reg = AlgorithmRegistry()
    reg.register("astar", AStarPlanner())
    return reg


def _agent_with_mock(response: str) -> SelectorAgent:
    agent = SelectorAgent.__new__(SelectorAgent)
    agent.name = "selector"
    agent.prompt_builder = SelectorPromptBuilder()
    agent.registry = _registry()
    agent.client = MagicMock()
    agent.client.generate.return_value = LLMResponse(response, "mock")
    return agent


class TestAlgorithmRegistry:
    def test_register_and_get(self):
        reg = AlgorithmRegistry()
        reg.register("astar", AStarPlanner())
        assert reg.get("astar") is not None
        assert "astar" in reg.names()

    def test_descriptions_returns_strings(self):
        reg = AlgorithmRegistry.default()
        for name, desc in reg.descriptions().items():
            assert isinstance(desc, str)

    def test_unknown_returns_none(self):
        assert AlgorithmRegistry().get("nonexistent") is None


class TestSelectorPrompt:
    def test_prompt_contains_conditions(self):
        pb = SelectorPromptBuilder()
        prompt = pb.build({"astar": "Classic A*"}, {"wind": "high"})
        assert "wind" in prompt
        assert "astar" in prompt


class TestSelectorAgent:
    def test_selects_and_runs_astar(self):
        agent = _agent_with_mock("Selected: astar\nReasoning: Best for low-density maps.")
        result = agent.run(_ctx())
        assert result.selected_algorithm == "astar"
        assert result.success
        assert result.reasoning

    def test_unknown_selection_falls_back_to_default(self):
        agent = _agent_with_mock("Selected: rrt\nReasoning: Fast in open space.")
        result = agent.run(_ctx())
        assert result.selected_algorithm == "astar"
        assert result.success

    def test_llm_error_uses_default(self):
        agent = SelectorAgent.__new__(SelectorAgent)
        agent.name = "selector"
        agent.prompt_builder = SelectorPromptBuilder()
        agent.registry = _registry()
        agent.client = MagicMock()
        agent.client.generate.side_effect = RuntimeError("offline")
        result = agent.run(_ctx())
        assert result.selected_algorithm == "astar"
        assert result.success
