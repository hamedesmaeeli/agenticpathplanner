"""CEC2024 benchmark loader — built-in maps + on-disk .npy files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from agenticpathplanner.core.types import MapInstance
from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS

log = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).parents[8] / "data" / "cec2024" / "maps"


class CEC2024Loader:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    def load(self, map_id: str) -> MapInstance:
        if map_id in BUILTIN_MAPS:
            return BUILTIN_MAPS[map_id]
        return self._load_from_disk(map_id)

    def load_all(self, difficulty: str | None = None, include_builtin: bool = True) -> list[MapInstance]:
        instances: list[MapInstance] = []
        if include_builtin:
            for m in BUILTIN_MAPS.values():
                if difficulty is None or m.difficulty == difficulty:
                    instances.append(m)
        for npy in sorted(self.data_dir.glob("*.npy")):
            if npy.stem in BUILTIN_MAPS:
                continue
            try:
                m = self._load_from_disk(npy.stem)
                if difficulty is None or m.difficulty == difficulty:
                    instances.append(m)
            except Exception as exc:
                log.warning("Skipping %s: %s", npy.stem, exc)
        return instances

    def iter_by_difficulty(self) -> Iterator[tuple[str, list[MapInstance]]]:
        for diff in ("simple", "medium", "complex"):
            batch = self.load_all(difficulty=diff)
            if batch:
                yield diff, batch

    def available_ids(self) -> list[str]:
        ids = list(BUILTIN_MAPS.keys())
        for npy in sorted(self.data_dir.glob("*.npy")):
            if npy.stem not in ids:
                ids.append(npy.stem)
        return ids

    def _load_from_disk(self, map_id: str) -> MapInstance:
        npy_path = self.data_dir / f"{map_id}.npy"
        meta_path = self.data_dir / f"{map_id}.yaml"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Map '{map_id}' not found at {npy_path}.\n"
                "Place official CEC2024 .npy files in data/cec2024/maps/."
            )
        grid = np.load(npy_path).astype(np.uint8)
        meta: dict = {}
        if meta_path.exists():
            with meta_path.open() as fh:
                meta = yaml.safe_load(fh) or {}
        return MapInstance(
            map_id=map_id,
            grid=grid,
            start=tuple(meta.get("start", [0, 0])),
            goal=tuple(meta.get("goal", [grid.shape[0] - 1, grid.shape[1] - 1])),
            difficulty=meta.get("difficulty", "medium"),
            description=meta.get("description", ""),
            metadata=meta,
        )
