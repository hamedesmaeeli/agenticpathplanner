"""agenticpathplanner.selector — Idea 3: LLM as Algorithm Selector.

The SelectorAgent maintains a registry of available planners. Given real-time
conditions (obstacle density, battery level, target movement, etc.), it asks
an LLM to reason about which algorithm is best suited, then executes it.
"""

from agenticpathplanner.selector.agent import SelectorAgent
from agenticpathplanner.selector.registry import AlgorithmRegistry
from agenticpathplanner.selector.prompts import SelectorPromptBuilder
from agenticpathplanner.selector.baselines import (
    RandomSelector, RuleBasedSelector, OracleSelector,
)

__all__ = [
    "SelectorAgent", "AlgorithmRegistry", "SelectorPromptBuilder",
    "RandomSelector", "RuleBasedSelector", "OracleSelector",
]
