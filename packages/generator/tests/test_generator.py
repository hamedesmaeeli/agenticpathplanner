"""Tests for agenticpathplanner.generator (mocked LLM)."""

from unittest.mock import MagicMock

import pytest

from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
from agenticpathplanner.core.agent import AgentContext
from agenticpathplanner.generator.agent import GeneratorAgent
from agenticpathplanner.generator.executor import CodeExecutor
from agenticpathplanner.generator.prompts import GeneratorPromptBuilder
from agenticpathplanner.llm.client import LLMResponse


def _ctx(mission="Find shortest path."):
    return AgentContext(instance=BUILTIN_MAPS["cec2024_s01"], mission=mission)


class TestGeneratorPrompt:
    def test_prompt_contains_mission(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        pb = GeneratorPromptBuilder()
        prompt = pb.build(inst, "Search and rescue mission")
        assert "rescue" in prompt.lower()
        assert str(inst.start) in prompt

    def test_system_prompt_has_function_signature(self):
        assert "plan(grid, start, goal)" in GeneratorPromptBuilder().system_prompt()


class TestCodeExecutor:
    def test_valid_code_returns_path(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        code = "def plan(grid, start, goal):\n    return [start, goal]"
        executor = CodeExecutor()
        path, stored = executor.execute(code, inst)
        assert path == [inst.start, inst.goal]
        assert stored is not None

    def test_markdown_fenced_python_code_returns_path(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        response = "```python\ndef plan(grid, start, goal):\n    return [start, goal]\n```"
        path, stored = CodeExecutor().execute(response, inst)
        assert path == [inst.start, inst.goal]
        assert stored is not None
        assert stored.startswith("def plan(")
        assert "```" not in stored

    def test_plain_python_code_is_preserved(self):
        code = "def plan(grid, start, goal):\n    return [start, goal]"
        assert CodeExecutor.extract_code(code) == code

    def test_unclosed_markdown_fence_is_not_silently_trimmed(self):
        response = "```python\ndef plan(grid, start, goal):\n    return [start, goal]"
        assert CodeExecutor.extract_code(response) == response

    def test_invalid_code_returns_empty(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        path, stored = CodeExecutor().execute("this is not python!!!", inst)
        assert path == []
        assert stored is None

    def test_missing_plan_fn_returns_empty(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        path, _ = CodeExecutor().execute("x = 42", inst)
        assert path == []


class TestGeneratorAgent:
    def _agent_with_mock(self, code: str) -> GeneratorAgent:
        agent = GeneratorAgent.__new__(GeneratorAgent)
        agent.name = "generator"
        agent.prompt_builder = GeneratorPromptBuilder()
        from agenticpathplanner.generator.executor import CodeExecutor
        agent.executor = CodeExecutor()
        agent.client = MagicMock()
        agent.client.generate.return_value = LLMResponse(code, "mock")
        return agent

    def test_valid_generated_code_produces_path(self):
        inst = BUILTIN_MAPS["cec2024_s01"]
        code = "def plan(grid, start, goal):\n    return [start, (4,4), goal]"
        agent = self._agent_with_mock(code)
        result = agent.run(_ctx())
        assert result.path[0] == inst.start
        assert result.path[-1] == inst.goal
        assert result.generated_code is not None

    def test_llm_error_returns_empty_path(self):
        agent = GeneratorAgent.__new__(GeneratorAgent)
        agent.name = "generator"
        agent.prompt_builder = GeneratorPromptBuilder()
        from agenticpathplanner.generator.executor import CodeExecutor
        agent.executor = CodeExecutor()
        agent.client = MagicMock()
        agent.client.generate.side_effect = RuntimeError("offline")
        result = agent.run(_ctx())
        assert not result.success
