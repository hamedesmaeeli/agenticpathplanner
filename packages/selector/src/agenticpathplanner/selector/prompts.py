"""Prompt templates for the Selector agent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a strategic path planning advisor for autonomous systems.

You will be given a set of available planning algorithms and real-time
environmental/operational conditions. Your task is to select the single
best algorithm for the current situation and explain your reasoning in
one concise sentence.

Output format (strictly follow this):
Selected: <algorithm_name>
Reasoning: <one sentence>

Where <algorithm_name> must be exactly one of the available algorithm names.
"""


class SelectorPromptBuilder:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build(self, algorithm_descriptions: dict[str, str], conditions: dict) -> str:
        algo_lines = "\n".join(
            f"  - {name}: {desc}" for name, desc in algorithm_descriptions.items()
        )
        cond_lines = "\n".join(f"  - {k}: {v}" for k, v in conditions.items()) or "  (none)"

        return (
            f"Available algorithms:\n{algo_lines}\n"
            f"\nCurrent conditions:\n{cond_lines}\n"
            f"\nSelect the best algorithm and explain why in one sentence."
        )
