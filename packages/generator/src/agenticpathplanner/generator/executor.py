"""Sandboxed Python code executor for LLM-generated planning functions."""

from __future__ import annotations

import logging
import textwrap

import numpy as np

from agenticpathplanner.core.types import MapInstance, Path

log = logging.getLogger(__name__)


class CodeExecutor:
    """Executes LLM-generated `plan(grid, start, goal)` code in a restricted namespace."""

    def execute(self, code: str, instance: MapInstance) -> tuple[Path, str | None]:
        """Run the generated code and return (path, generated_code_string)."""
        code = code.strip()
        if not code:
            return [], None

        import math as _m, heapq as _h, collections as _c
        namespace: dict = {
            "numpy": np, "np": np,
            "math": _m, "heapq": _h, "collections": _c,
            "__builtins__": {k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
                             if k in ("range","len","list","tuple","int","float","bool",
                                      "min","max","abs","sum","sorted","enumerate","zip",
                                      "print","isinstance","type","hasattr","__import__")},
        }
        try:
            exec(textwrap.dedent(code), namespace)
        except Exception as exc:
            log.warning("Generated code failed to compile: %s", exc)
            return [], None

        plan_fn = namespace.get("plan")
        if not callable(plan_fn):
            log.warning("Generated code has no callable `plan` function.")
            return [], None

        try:
            result = plan_fn(instance.grid, instance.start, instance.goal)
            path = [(int(x[0]), int(x[1])) for x in result
                    if isinstance(x, (list, tuple)) and len(x) == 2]
            return path, code
        except Exception as exc:
            log.warning("Generated plan() raised: %s", exc)
            return [], None
