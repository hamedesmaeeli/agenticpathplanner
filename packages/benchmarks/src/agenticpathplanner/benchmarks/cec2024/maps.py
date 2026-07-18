"""CEC2024 built-in map instances.

MapInstance is defined in agenticpathplanner.core so all subprojects share the
same type.  This module supplies the concrete benchmark instances.
"""

from __future__ import annotations

import numpy as np

from agenticpathplanner.core.types import MapInstance


def _make_simple_map() -> np.ndarray:
    g = np.zeros((10, 10), dtype=np.uint8)
    g[5, 1:4] = 1
    g[5, 6:9] = 1
    return g


def _make_medium_map() -> np.ndarray:
    g = np.zeros((20, 20), dtype=np.uint8)
    g[4, 2:14] = 1
    g[8, 6:18] = 1
    g[12, 2:10] = 1
    g[16, 8:18] = 1
    g[2:8, 14] = 1
    g[10:16, 4] = 1
    return g


def _make_complex_map() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    g = np.zeros((30, 30), dtype=np.uint8)
    for r in range(30):
        for c in range(30):
            if rng.random() < 0.28:
                g[r, c] = 1
    g[1, 1] = 0
    g[28, 28] = 0
    return g


BUILTIN_MAPS: dict[str, MapInstance] = {
    "cec2024_s01": MapInstance(
        "cec2024_s01", _make_simple_map(), (1, 1), (8, 8),
        difficulty="simple", description="10×10 open room with central wall gap",
    ),
    "cec2024_m01": MapInstance(
        "cec2024_m01", _make_medium_map(), (1, 1), (18, 18),
        difficulty="medium", description="20×20 corridor maze",
    ),
    "cec2024_c01": MapInstance(
        "cec2024_c01", _make_complex_map(), (1, 1), (28, 28),
        difficulty="complex", description="30×30 randomly cluttered environment",
    ),
}
